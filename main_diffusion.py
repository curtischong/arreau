import argparse
import os
from diffusion.lattice_dataset import CrystalDataset
from lightning_wrappers.diffusion import PONITA_DIFFUSION
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph
import pytorch_lightning as pl
from lightning_wrappers.callbacks import EMA, EpochTimer


# ------------------------ Function to convert the nbody dataset to a dataloader for pytorch geometric graphs

def make_pyg_loader(dataset, batch_size, shuffle, num_workers, radius, loop):
    data_list = []
    radius = radius or 1000.
    radius_graph = RadiusGraph(radius, loop=loop, max_num_neighbors=1000)
    for data in dataset:
        loc, vel, edge_attr, charges, loc_end = data
        x = charges
        vec = vel[:,None,:]  # [num_pts, num_channels=1, 3]
        # Build the graph
        graph = Data(pos=loc, x=x, vec=vec, y=loc_end)
        graph = radius_graph(graph)
        # Append to the database list
        data_list.append(graph)
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# ------------------------ Start of the main experiment script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ------------------------ Input arguments

    # Run parameters
    parser.add_argument('--epochs', type=int, default=10000,
                        help='number of epochs')
    parser.add_argument('--warmup', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-10,
                        help='weight decay')
    parser.add_argument('--log', type=eval, default=True,
                        help='logging flag')
    parser.add_argument('--enable_progress_bar', type=eval, default=False,
                        help='enable progress bar')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Num workers in dataloader')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--val_interval', type=int, default=5, metavar='N',
                        help='how many epochs to wait before logging validation')
    
    # Train settings
    parser.add_argument('--train_augm', type=eval, default=True,
                        help='whether or not to use random rotations during training')
    
    # nbody Dataset
    parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                        help='maximum amount of training samples')
    parser.add_argument('--dataset', type=str, default="alexandria", metavar='N',
                        help='nbody_small, nbody')
    
    # Graph connectivity settings
    parser.add_argument('--radius', type=eval, default=None,
                        help='radius for the radius graph construction in front of the force loss')
    parser.add_argument('--loop', type=eval, default=True,
                        help='enable self interactions')
    
    # PONTA model settings
    parser.add_argument('--num_ori', type=int, default=16,
                        help='num elements of spherical grid')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='internal feature dimension')
    parser.add_argument('--basis_dim', type=int, default=256,
                        help='number of basis functions')
    parser.add_argument('--degree', type=int, default=3,
                        help='degree of the polynomial embedding')
    parser.add_argument('--layers', type=int, default=5,
                        help='Number of message passing layers')
    parser.add_argument('--widening_factor', type=int, default=4,
                        help='Number of message passing layers')
    parser.add_argument('--layer_scale', type=float, default=1e-6,
                        help='Initial layer scale factor in ConvNextBlock, 0 means do not use layer scale')
    parser.add_argument('--multiple_readouts', type=eval, default=True,
                        help='Whether or not to readout after every layer')
    parser.add_argument('--num_timesteps', type=int,
                        help='the number of diffusion timesteps')
    
    # Parallel computing stuff
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus to use (assumes all are on one node)')
    
    # Arg parser
    args = parser.parse_args()

    # ------------------------ Device settings

    if args.gpus > 0:
        accelerator = "gpu"
        devices = args.gpus
    else:
        accelerator = "cpu"
        devices = "auto"
    if args.num_workers == -1:
        args.num_workers = os.cpu_count()

    # ------------------------ Dataset

    train_dataset = CrystalDataset([
        "datasets/alexandria_hdf5/alexandria_ps_000.h5",
        "datasets/alexandria_hdf5/alexandria_ps_001.h5",
        "datasets/alexandria_hdf5/alexandria_ps_002.h5",
    ])
    valid_dataset = CrystalDataset([
        "datasets/alexandria_hdf5/alexandria_ps_003.h5",
    ])
    test_dataset = CrystalDataset([
        "datasets/alexandria_hdf5/alexandria_ps_004.h5",
    ])
    
    datasets = {'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset}

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
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == 'train'), num_workers=args.num_workers, persistent_workers=True)
        for split, dataset in datasets.items()}
    
    # ------------------------ Load and initialize the model

    model = PONITA_DIFFUSION(args, dataset.num_atomic_states)

    # ------------------------ Weights and Biases logger

    if args.log:
        logger = pl.loggers.WandbLogger(project="PONITA-" + args.dataset, name='siva', config=args, save_dir='logs')
    else:
        logger = None

    # ------------------------ Set up the trainer

    # Seed
    pl.seed_everything(args.seed, workers=True)
    
    # Pytorch lightning call backs
    callbacks = [EMA(0.99),
                pl.callbacks.ModelCheckpoint(monitor='valid loss', mode = 'min'),
                 EpochTimer()]
    if args.log: callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))

    # Initialize the trainer
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, callbacks=callbacks, gradient_clip_val=0.5, 
                         accelerator=accelerator, devices=devices, check_val_every_n_epoch=args.val_interval,
                         enable_progress_bar=args.enable_progress_bar)
                        #  log_every_n_steps=1) # TODO: increase this

    # Do the training
    trainer.fit(model, dataloaders['train'], dataloaders['valid'])

    # And test
    trainer.test(model, dataloaders['test'], ckpt_path = "best")
