from dataclasses import dataclass
import multiprocessing
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch
from torch_geometric.data import Data

from diffusion.tools.atomic_number_table import (
    get_atomic_number_table_from_zs,
    one_hot_encode_atomic_numbers,
)
from diffusion.tools.neighborhood import get_neighborhood


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
    # configs = [load_dataset(config_path) for config_path in config_paths]
    # return [item for sublist in configs for item in sublist]


# This dataset will not be good for larger systems since it loads all of the data into memory
# if we are to scale this for much larger datasets, we need to only load the hdf5 files during training
# We also need to precalculate the number of unique atomic numbers when generating the hdf5 files
# so we don't have to load all the data initially
class CrystalDataset(Dataset):
    def __init__(self, config_paths: list[str], cutoff: int = 5.0):
        # configs = load_dataset("datasets/alexandria_hdf5/10_examples.h5")
        # configs = load_dataset("datasets/alexandria_hdf5/alexandria_ps_000.h5")
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

    def get_cell_info(
        self, *, Xt: np.ndarray, Lt: np.ndarray
    ):  # config: Configuration | InferenceConfiguration, timestep):
        positions = Xt @ Lt
        pbc = [True, True, True]  # all true for an infinite lattice

        # I think cell = L is required so when we extend the lattice in infinite directions, are are tiling the cell properly
        edge_index, shifts, unit_shifts = get_neighborhood(
            positions=Xt, cutoff=self.cutoff, pbc=pbc, cell=Lt
        )
        # alternatve if the cell and position is wrong. But this will result in weird tiling of the cells
        # edge_index, shifts, unit_shifts = get_neighborhood(
        #     positions=config.X @ config.L, cutoff=cutoff, pbc=pbc, cell=None
        # )
        return {
            "edge_index": edge_index,
            "positions": positions,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
        }

    def __len__(self):
        return len(self.configs)

    def __getitem__(self, idx: int):
        config = self.configs[idx]
        A0 = one_hot_encode_atomic_numbers(self.z_table, config.atomic_numbers)

        X0 = config.X0
        L0 = config.L0  # TODO(curtis): use the noised Lt
        cell_info = self.get_cell_info(Xt=X0, Lt=L0)

        #  Data(pos=loc, x=x, vec=vec, y=loc_end)
        device = torch.cuda.current_device()
        return Data(
            pos=torch.tensor(
                cell_info["positions"],
                dtype=torch.get_default_dtype(),
                device=device,
            ),
            x=A0,  # These are the node features (that's why it's called x, not A0)
            # A0=A0,
            X0=torch.tensor(X0, dtype=torch.get_default_dtype(), device=device),
            L0=torch.tensor(L0, dtype=torch.get_default_dtype(), device=device),
            num_atoms=len(config.atomic_numbers),
        )
