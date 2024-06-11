from dataclasses import dataclass
import multiprocessing
from typing import List
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch
from torch_geometric.data import Data

from diffusion.d3pm import D3PM
from diffusion.diffusion_helpers import VE_pbc
from diffusion.tools.atomic_number_table import (
    atomic_numbers_to_indices,
    get_atomic_number_table_from_zs,
)


@dataclass
class Configuration:
    # fields with a prefix "prev_" are the real target value. These are optional since during inference, we don't know them
    atomic_numbers: np.ndarray  # this is a list of atomic numbers. NOT one-hot-encoded
    X0: np.ndarray
    L0: np.ndarray


def load_data(filepath: str):
    with h5py.File(filepath, "r") as file:
        frac_x = (file["crystals"]["frac_x"][:],)
        atomic_numbers = (file["crystals"]["atomic_numbers"][:],)
        lattice = (file["crystals"]["lattice"][:],)
        num_atoms = (file["crystals"]["num_atoms"][:],)
        idx_start = (file["crystals"]["idx_start"][:],)

    return frac_x[0], atomic_numbers[0], lattice[0], num_atoms[0], idx_start[0]


def load_dataset(file_path) -> List[Configuration]:
    frac_x, atomic_numbers, lattice, num_atoms, idx_start = load_data(file_path)

    dataset = []
    for i in range(lattice.shape[0]):
        start = idx_start[i]
        end = start + num_atoms[i]
        config = Configuration(
            atomic_numbers=atomic_numbers[start:end],
            X0=frac_x[start:end],
            L0=lattice[i],
        )
        dataset.append(config)
    return dataset


def parallelize_loading_configs(config_paths):
    with multiprocessing.Pool() as pool:
        configs = pool.map(load_dataset, config_paths)
        return [item for sublist in configs for item in sublist]
    # keep this here so we can use the single-threded version for debugging
    # configs = [load_dataset(config_path) for config_path in config_paths]
    # return [item for sublist in configs for item in sublist]


# This dataset will not be good for larger systems since it loads all of the data into memory
# if we are to scale this for much larger datasets, we need to only load the hdf5 files during training
# We also need to precalculate the number of unique atomic numbers when generating the hdf5 files
# so we don't have to load all the data initially
class AlexandriaDataset(Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.unique_atomic_numbers = set()

        self.configs = load_dataset(dataset_path)
        for config in self.configs:
            self.unique_atomic_numbers.update(set(np.asarray(config.atomic_numbers)))

        self.z_table = get_atomic_number_table_from_zs(
            [
                self.unique_atomic_numbers,
            ]
        )
        print(f"There are {len(self.z_table)} unique atomic numbers")

        print(
            f"finished loading datasets {str(dataset_path)}. Found {len(self.configs)} entries"
        )
        # self.num_timesteps = model_attributes["num_timesteps"]

    def num_classes(self):
        return len(self.z_table)

    def set_diffusion_modules(self, x_frac_diffusion: VE_pbc, atomic_diffusion: D3PM):
        self.x_frac_diffusion = x_frac_diffusion
        self.atomic_diffusion = atomic_diffusion

    def __len__(self):
        return len(self.configs)

    def __getitem__(self, idx: int):
        # default_dtype = torch.float64  # for some reason, the default dtype is float32 in this subprocess. so I set it explicitly
        default_dtype = torch.float32  # Doing this because equiformer uses float32 linear layers. I don't know why. But if I have precision issues, I'll probably change this. The only reason why I'm okay with float32 is because we're not doing molecular dynamics
        config = self.configs[idx]

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print("NOTE: all of the data is only on gpu0")
        # device = torch.device(f"cuda:{0}")
        device = torch.device("cpu")

        # Get the initial state
        x_frac_start = torch.tensor(config.X0, dtype=default_dtype, device=device)
        atom_type_start = atomic_numbers_to_indices(
            self.z_table, config.atomic_numbers
        ).to(device)
        cell_start = torch.tensor(config.L0, dtype=default_dtype, device=device).view(
            -1, 3, 3
        )

        # # get the noisy state
        # timestep = torch.randint(1, self.num_timesteps + 1, size=(1,), device=device)

        # atom_type_noisy = self.atomic_diffusion.get_xt(atom_type_start, timestep)
        # x_frac_noisy, _wrapped_frac_eps_x, _used_sigmas = self.x_frac_diffusion.forward(
        #     x_frac_start,
        #     timestep.unsqueeze(0),
        #     cell_start.unsqueeze(0),
        #     torch.tensor(len(x_frac_start), dtype=torch.int, device=device).unsqueeze(
        #         0
        #     ),
        # )

        # x_cart_noisy = (x_frac_start @ cell_start).clone().detach().to(device)
        # # NOTE: we cannot fix the features at this point because when we add noise, the features will change

        res = Data(
            # x_cart_noisy=x_cart_noisy,
            X0=x_frac_start,
            # x_frac_noisy=x_frac_noisy,
            A0=atom_type_start,
            # atom_type_noisy=atom_type_noisy,
            L0=cell_start,
            natoms=torch.tensor(len(config.atomic_numbers), device=device),
            ith_sample=torch.tensor(idx, dtype=torch.float32),
            # timestep=timestep,
        )
        return res
