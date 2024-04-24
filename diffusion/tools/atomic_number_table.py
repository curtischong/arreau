import numpy as np
from typing import Sequence
import torch
from pymatgen.core import Element


class AtomicNumberTable:
    MASK_ATOMIC_NUMBER = (
        2001  # This is the mask atomic number used in the mattergen paper (from d3pm)
    )

    def __init__(self, zs: Sequence[int]):
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self):
        return f"AtomicNumberTable: {tuple(s for s in self.zs)}"

    def index_to_z(self, index: int) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: int) -> int:
        return self.zs.index(atomic_number)


def get_atomic_number_table_from_zs(zs: set[int]) -> AtomicNumberTable:
    z_set = set(zs[0])  # copy the original set so we don't modify it
    for i in range(1, len(zs)):
        z_set.update(zs[i])
    z_set.add(AtomicNumberTable.MASK_ATOMIC_NUMBER)
    return AtomicNumberTable(sorted(list(z_set)))


def atomic_numbers_to_indices(
    z_table: AtomicNumberTable,
    atomic_numbers: np.ndarray,
) -> torch.Tensor:
    to_index_fn = np.vectorize(z_table.z_to_index)
    atomic_number_indices = to_index_fn(atomic_numbers)
    return torch.tensor(atomic_number_indices, dtype=torch.long)


def atomic_number_indexes_to_atomic_numbers(
    z_table: AtomicNumberTable,
    atomic_number_indexes: np.ndarray,
) -> np.ndarray:
    to_atomic_num = np.vectorize(z_table.index_to_z)
    return to_atomic_num(atomic_number_indexes)


def one_hot_to_atomic_numbers(
    z_table: AtomicNumberTable, one_hot: torch.Tensor
) -> np.ndarray:
    atomic_number_indexes = one_hot.argmax(dim=1).numpy()
    return atomic_number_indexes_to_atomic_numbers(z_table, atomic_number_indexes)


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


def one_hot_encode_atomic_numbers(
    z_table: AtomicNumberTable, atomic_numbers: np.ndarray
) -> np.ndarray:
    atomic_number_indices = atomic_numbers_to_indices(z_table, atomic_numbers)
    return to_one_hot(atomic_number_indices.unsqueeze(-1), num_classes=len(z_table))


def atomic_symbols_to_indices(
    z_table: AtomicNumberTable,
    atomic_symbols: list[str],
) -> np.ndarray:
    atomic_numbers = [Element(symbol).number for symbol in atomic_symbols]
    return atomic_numbers_to_indices(z_table, atomic_numbers)
