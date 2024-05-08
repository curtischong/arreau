import argparse
import os
from pathlib import Path
from diffusion.lattice_dataset import CrystalDataset
from lightning_wrappers.diffusion import PONITA_DIFFUSION
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph
import pytorch_lightning as pl
from lightning_wrappers.callbacks import EpochTimer
import torch
from pytorch_lightning.profilers import PyTorchProfiler


# ------------------------ Function to convert the nbody dataset to a dataloader for pytorch geometric graphs


def make_pyg_loader(dataset, batch_size, shuffle, num_workers, radius, loop):
    data_list = []
    radius = radius or 1000.0
    radius_graph = RadiusGraph(radius, loop=loop, max_num_neighbors=1000)
    for data in dataset:
        loc, vel, edge_attr, charges, loc_end = data
        x = charges
        vec = vel[:, None, :]  # [num_pts, num_channels=1, 3]
        # Build the graph
        graph = Data(pos=loc, x=x, vec=vec, y=loc_end)
        graph = radius_graph(graph)
        # Append to the database list
        data_list.append(graph)
    return DataLoader(
        data_list, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def get_active_branch_name():
    head_dir = Path(".") / ".git" / "HEAD"
    with head_dir.open("r") as f:
        content = f.read().splitlines()

    for line in content:
        if line[0:4] == "ref:":
            return line.partition("refs/heads/")[2]


# ------------------------ Start of the main experiment script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ------------------------ Input arguments

    # Run parameters
    parser.add_argument("--epochs", type=int, default=10000, help="number of epochs")
    parser.add_argument("--warmup", type=int, default=10, help="number of epochs")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size. Does not scale with number of gpus.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-10, help="weight decay"
    )
    parser.add_argument("--log", type=eval, default=True, help="logging flag")
    parser.add_argument(
        "--enable_progress_bar", type=eval, default=False, help="enable progress bar"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Num workers in dataloader"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--val_interval",
        type=int,
        default=5,
        metavar="N",
        help="how many epochs to wait before logging validation",
    )

    # Train settings
    parser.add_argument(
        "--train_augm",
        type=eval,
        default=False,
        help="whether or not to use random rotations during training",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="alexandria",
    )

    # Graph connectivity settings
    parser.add_argument(
        "--radius",
        type=eval,
        default=None,
        help="radius for the radius graph construction in front of the force loss",
    )
    parser.add_argument(
        "--loop", type=eval, default=True, help="enable self interactions"
    )

    # PONTA model settings
    parser.add_argument(
        "--num_ori", type=int, default=16, help="num elements of spherical grid"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="internal feature dimension"
    )
    parser.add_argument(
        "--basis_dim", type=int, default=256, help="number of basis functions"
    )
    parser.add_argument(
        "--degree", type=int, default=3, help="degree of the polynomial embedding"
    )
    parser.add_argument(
        "--layers", type=int, default=5, help="Number of message passing layers"
    )
    parser.add_argument(
        "--widening_factor",
        type=int,
        default=4,
        help="Number of message passing layers",
    )
    parser.add_argument(
        "--layer_scale",
        type=float,
        default=1e-6,
        help="Initial layer scale factor in ConvNextBlock, 0 means do not use layer scale",
    )
    parser.add_argument(
        "--multiple_readouts",
        type=eval,
        default=True,
        help="Whether or not to readout after every layer",
    )
    parser.add_argument(
        "--num_timesteps", type=int, help="the number of diffusion timesteps"
    )
    parser.add_argument(
        "--max_neighbors",
        type=int,
        required=True,
        help="the maximum number of other atoms an atom can be directly influenced by",
    )
    parser.add_argument(
        "--experiment_name", type=str, help="the number of diffusion timesteps"
    )
    parser.add_argument(
        "--profiler",
        type=str,
        default=False,
        help="Specifies the type of profiler",
        choices=["pytorch", "advanced"],
    )

    # Parallel computing stuff
    parser.add_argument(
        "-g",
        "--gpus",
        default=1,
        type=int,
        help="number of gpus to use (assumes all are on one node)",
    )

    # Arg parser
    args = parser.parse_args()

    # ------------------------ Device settings

    if args.gpus > 0:
        accelerator = "gpu"
        devices = args.gpus
        # torch.set_default_device("cuda:0")
    else:
        accelerator = "cpu"
        devices = "auto"
    if args.num_workers == -1:
        args.num_workers = os.cpu_count()
    torch.set_default_dtype(torch.float64)

    # ------------------------ Dataset

    def get_default_device():
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    # TODO: remove this if statement and put it all into a config
    if args.dataset == "alexandria-dev":
        print("Using dev dataset")
        dataset = CrystalDataset(
            [
                "datasets/alexandria_hdf5/alexandria_ps_000_take10.h5",
            ]
        )
        train_dataset = dataset
        valid_dataset = dataset
        test_dataset = dataset
        z_table = train_dataset.z_table
    elif args.dataset == "eval-equivariance":
        train_dataset = CrystalDataset(
            [
                "datasets/alexandria_hdf5/alexandria_ps_000_take1.h5",
            ]
        )
        valid_dataset = CrystalDataset(
            [
                "datasets/alexandria_hdf5/alexandria_ps_000_take1_rotated.h5",
            ]
        )
        test_dataset = valid_dataset
        z_table = train_dataset.z_table
    else:
        dataset = CrystalDataset(
            [
                "datasets/alexandria_hdf5/alexandria_ps_000.h5",
                "datasets/alexandria_hdf5/alexandria_ps_001.h5",
                "datasets/alexandria_hdf5/alexandria_ps_002.h5",
                "datasets/alexandria_hdf5/alexandria_ps_003.h5",
                "datasets/alexandria_hdf5/alexandria_ps_004.h5",
            ]
        )
        z_table = dataset.z_table

        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [0.7, 0.15, 0.15],
            # generator=torch.Generator(device=get_default_device()),
        )

    datasets = {"train": train_dataset, "valid": valid_dataset, "test": test_dataset}

    # TODO: look into using this make_pgy_loader, since it automatically attaches edges. We are not using it here since we purturb the location of the atoms later.
    # if we are using it, we should use it there

    # dataloaders = {
    #     split: make_pyg_loader(dataset,
    #                            batch_size=args.batch_size,
    #                            shuffle=(split == 'train'),
    #                            num_workers=args.num_workers,
    #                            radius=args.radius,
    #                            loop=args.loop)
    #     for split, dataset in datasets.items()}

    # Make the dataloaders
    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            persistent_workers=True,
        )
        for split, dataset in datasets.items()
    }

    # ------------------------ Load and initialize the model

    model = PONITA_DIFFUSION(args, z_table)

    # ------------------------ Weights and Biases logger

    if args.experiment_name is None:
        args.experiment_name = get_active_branch_name()
        if args.dataset == "alexandria-dev":
            args.experiment_name = "local-" + args.experiment_name
        elif args.dataset == "eval-equivariance":
            args.experiment_name = "eval-equivariance-" + args.experiment_name

    if args.log:
        if not args.experiment_name:
            raise ValueError("You need to specify an experiment name")
        logger = pl.loggers.WandbLogger(
            project="PONITA-alexandria",
            name=args.experiment_name,
            config=args,
            save_dir="logs",
        )
    else:
        logger = None

    # ------------------------ Set up the trainer

    # Seed
    pl.seed_everything(args.seed, workers=True)

    # Pytorch lightning call backs
    callbacks = []
    # if args.dataset != "eval-equivariance":
    #     callbacks.append(
    #         EMA(0.99)
    #     )  # disable this for eval-equivariance so the train and validation loss matches
    callbacks += [
        pl.callbacks.ModelCheckpoint(
            dirpath="checkpoints",
            filename="model-{epoch:02d}-{valid_loss:.2f}",
            monitor="valid loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EpochTimer(),
    ]
    if args.log:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="epoch"))

    if args.profiler == "pytorch":
        profiler = PyTorchProfiler(
            dirpath="profile_results",
            row_limit=None,
        )
    elif args.profiler == "advanced":
        profiler = "advanced"
    else:
        profiler = None

    # Initialize the trainer
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=callbacks,
        gradient_clip_val=0.5,
        accelerator=accelerator,
        devices=devices,
        check_val_every_n_epoch=args.val_interval,
        enable_progress_bar=args.enable_progress_bar,
        profiler=profiler,
    )
    #  log_every_n_steps=1) # TODO: increase this

    # Do the training
    trainer.fit(model, dataloaders["train"], dataloaders["valid"])

    # And test
    trainer.test(model, dataloaders["test"], ckpt_path="best")
