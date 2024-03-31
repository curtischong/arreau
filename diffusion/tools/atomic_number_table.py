import numpy as np
from typing import Iterable, Sequence
import torch

class AtomicNumberTable:
    MASK_ATOMIC_NUMBER = 2001 # This is the mask atomic number used in the mattergen paper

    def __init__(self, zs: Sequence[int]):
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self):
        return f"AtomicNumberTable: {tuple(s for s in self.zs)}"

    def index_to_z(self, index: int) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: str) -> int:
        return self.zs.index(atomic_number)
    

def get_atomic_number_table_from_zs(zs: set[int]) -> AtomicNumberTable:
    z_set = set(zs[0]) # copy the original set so we don't modify it
    for i in range(1, len(zs)):
        z_set.update(zs[i])
    # z_set.add(AtomicNumberTable.MASK_ATOMIC_NUMBER)
    return AtomicNumberTable(sorted(list(z_set)))

def atomic_numbers_to_indices(
    atomic_numbers: np.ndarray, z_table: AtomicNumberTable
) -> np.ndarray:
    to_index_fn = np.vectorize(z_table.z_to_index)
    return to_index_fn(atomic_numbers)

def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Generates one-hot encoding with <num_classes> classes from <indices>
    :param indices: (N x 1) tensor
    :param num_classes: number of classes
    :param device: torch device
    :return: (N x num_classes) tensor
    """
    shape = indices.shape[:-1] + (num_classes,)
    oh = torch.zeros(shape, device=indices.device).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)