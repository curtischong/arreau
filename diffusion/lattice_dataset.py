from dataclasses import dataclass
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch
from diffusion.atomic_data import AtomicData

from diffusion.tools.atomic_number_table import atomic_numbers_to_indices, get_atomic_number_table_from_zs, to_one_hot
from diffusion.tools.neighborhood import get_neighborhood


@dataclass
class Configuration:
    # fields with a prefix "prev_" are the real target value. These are optional since during inference, we don't know them
    atomic_numbers: np.ndarray # this is a list of atomic numbers. NOT one-hot-encoded
    X0: np.ndarray
    L0: np.ndarray

def load_data(filename):
    with h5py.File(filename, 'r') as f:
        # Load atom one-hot matrices
        atomic_number_vectors = []
        for key in sorted(f['atomic_number'], key=lambda x: int(x)):
            atomic_number_vectors.append(np.array(f['atomic_number'][key]))

        # Load lattice matrices
        lattice_matrices = np.array(f['lattice_matrix'])

        # Load fractional coordinates arrays
        frac_coords_arrays = []
        for key in sorted(f['frac_coord'], key=lambda x: int(x)):
            frac_coords_arrays.append(np.array(f['frac_coord'][key]))

    return atomic_number_vectors, lattice_matrices, frac_coords_arrays

# @gin.configurable
def load_dataset(file_path) -> list[Configuration]:
    atomic_number_vector, lattice_matrix, frac_coord = load_data(file_path)

    dataset = []
    for i in range(len(lattice_matrix)):
        assert lattice_matrix[i].shape == (3, 3)
        config = Configuration(
            atomic_numbers = atomic_number_vector[i],
            X0 = frac_coord[i],
            L0 = lattice_matrix[i],
        )
        dataset.append(config)
    print(f"num training samples: {len(dataset)}")
    # for i in range(len(dataset)):
    #     print(i, dataset[i].atomic_numbers)
    # return dataset[:4]
    # return dataset[:5]
    return dataset

class CrystalDataset(Dataset):
    def __init__(self, cutoff: int = 5.0):
        configs = load_dataset("datasets/alexandria_hdf5/10_examples.h5")
        # configs = load_dataset("datasets/alexandria_hdf5/alexandria_ps_000.h5")
        z_table = get_atomic_number_table_from_zs(
            z
            for config in configs
            for z in config.atomic_numbers
        )
        self.configs = configs
        self.z_table = z_table
        self.cutoff = cutoff
        self.num_atomic_states = len(z_table)
        # print("finished loading dataset")

    def num_atomic_states(self):
        return len(self.z_table)

    # maybe we should move this to utils? oh. it does depend on self.z_table though
    def one_hot_encode_atomic_numbers(self, atomic_numbers: np.ndarray) -> np.ndarray:
        atomic_number_indices = atomic_numbers_to_indices(atomic_numbers, z_table=self.z_table)
        atomic_number_indices_torch = torch.tensor(atomic_number_indices, dtype=torch.long)
        A0 = to_one_hot(
                atomic_number_indices_torch.unsqueeze(-1),
                num_classes=self.num_atomic_states
            )
        return A0, atomic_number_indices_torch

    def get_cell_info(self, *, Xt: np.ndarray, Lt: np.ndarray):#config: Configuration | InferenceConfiguration, timestep):
        positions = Xt @ Lt
        pbc = [True, True, True] # all true for an infinite lattice

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
        A0, _atomic_number_indices_torch = self.one_hot_encode_atomic_numbers(config.atomic_numbers)

        X0 = config.X0
        L0 = config.L0 # TODO(curtis): use the noised Lt
        cell_info = self.get_cell_info(Xt=X0, Lt=L0)

        return AtomicData(
            edge_index=torch.tensor(cell_info["edge_index"], dtype=torch.long),
            positions=torch.tensor(cell_info["positions"], dtype=torch.get_default_dtype()),
            shifts=torch.tensor(cell_info["shifts"], dtype=torch.get_default_dtype()),
            unit_shifts=torch.tensor(cell_info["unit_shifts"], dtype=torch.get_default_dtype()),
            A0=A0,
            X0=torch.tensor(X0, dtype=torch.get_default_dtype()),
            L0=torch.tensor(L0, dtype=torch.get_default_dtype()),
        )