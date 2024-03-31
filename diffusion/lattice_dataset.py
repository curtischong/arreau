from dataclasses import dataclass
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch
from diffusion.atomic_data import AtomicData
from torch_geometric.data import Data

from diffusion.tools.atomic_number_table import AtomicNumberTable, atomic_numbers_to_indices, get_atomic_number_table_from_zs, to_one_hot
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
    return dataset

class CrystalDataset(Dataset):
    def __init__(self, config_paths: list[str], cutoff: int = 5.0):
        # configs = load_dataset("datasets/alexandria_hdf5/10_examples.h5")
        # configs = load_dataset("datasets/alexandria_hdf5/alexandria_ps_000.h5")
        self.unique_atomic_numbers = set()
        configs = []
        for config_path in config_paths:
            configs += load_dataset(config_path)
        for config in configs:
            self.unique_atomic_numbers.update(set(config.atomic_numbers))
        self.configs = configs
        self.cutoff = cutoff
        print(f"finished loading datasets {str(config_paths)}. Found {len(self.configs)} entries")

    def set_z_table(self, z_table: AtomicNumberTable):
        self.z_table = z_table
        self.num_atomic_states = len(z_table)

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

        #  Data(pos=loc, x=x, vec=vec, y=loc_end)
        return Data(
            pos=torch.tensor(cell_info["positions"], dtype=torch.get_default_dtype()),
            x=A0, # These are the node features (that's why it's called x, not A0)
            # A0=A0,
            X0=torch.tensor(X0, dtype=torch.get_default_dtype()),
            L0=torch.tensor(L0, dtype=torch.get_default_dtype()),
            num_atoms=len(config.atomic_numbers),
        )