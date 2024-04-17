from dataclasses import dataclass
import multiprocessing
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch
from torch_geometric.data import Data

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


def load_data(filename: str):
    with h5py.File(filename, "r") as f:
        # Load atom one-hot matrices
        sorted_keys = sorted(f["atomic_number"], key=int)
        atomic_number_vectors = [None] * len(sorted_keys)
        for i in range(len(sorted_keys)):
            key = sorted_keys[i]
            atomic_number_vectors[i] = np.array(f["atomic_number"][key])

        # Load lattice matrices
        lattice_matrices = np.array(f["lattice_matrix"])

        # Load fractional coordinates arrays
        sorted_keys = sorted(f["frac_coord"], key=int)
        frac_coords_arrays = [None] * len(sorted_keys)
        for i in range(len(sorted_keys)):
            key = sorted_keys[i]
            frac_coords_arrays[i] = np.array(f["frac_coord"][key])

    return atomic_number_vectors, lattice_matrices, frac_coords_arrays


def load_dataset(file_path) -> list[Configuration]:
    atomic_number_vector, lattice_matrix, frac_coord = load_data(file_path)

    dataset = []
    for i in range(len(lattice_matrix)):
        assert lattice_matrix[i].shape == (3, 3)
        config = Configuration(
            atomic_numbers=atomic_number_vector[i],
            X0=frac_coord[i],
            L0=lattice_matrix[i],
        )
        dataset.append(config)
    return dataset


def parallelize_configs(config_paths):
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
class CrystalDataset(Dataset):
    def __init__(self, config_paths: list[str], cutoff: int = 5.0):
        self.unique_atomic_numbers = set()
        configs = parallelize_configs(config_paths)
        for config in configs:
            self.unique_atomic_numbers.update(set(config.atomic_numbers))
        self.configs = configs
        self.cutoff = cutoff

        self.z_table = get_atomic_number_table_from_zs(
            [
                self.unique_atomic_numbers,
            ]
        )
        print(f"There are {len(self.z_table)} unique atomic numbers")

        print(
            f"finished loading datasets {str(config_paths)}. Found {len(self.configs)} entries"
        )

    def __len__(self):
        return len(self.configs)

    def __getitem__(self, idx: int):
        config = self.configs[idx]
        A0 = atomic_numbers_to_indices(self.z_table, config.atomic_numbers)
        X0 = config.X0
        L0 = config.L0

        X0_cart = torch.tensor(
            X0 @ L0,
            dtype=torch.get_default_dtype(),
        )
        return Data(
            pos=X0_cart,  # we need to have a pos field so the datalolader generates the batch atribute for each batch
            X0=torch.tensor(X0, dtype=torch.get_default_dtype()),
            A0=A0,
            L0=torch.tensor(L0, dtype=torch.get_default_dtype()),
            num_atoms=len(config.atomic_numbers),
        )
