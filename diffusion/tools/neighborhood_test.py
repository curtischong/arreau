from diffusion.tools.neighborhood import (
    get_neighborhood_for_batch,
    atom_cart_coords_in_supercell,
)
import torch


def test_get_neighborhood():
    # frac_coords = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.5, 0.5, 0.5]])
    torch.set_default_dtype(torch.float64)
    frac_coords = torch.tensor(
        [[0.5, 0.5, 0.5], [0.5, 0.5, 0.6], [0.5, 0.8, 0.8]],
        dtype=torch.get_default_dtype(),
    )
    lattice = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[10.0, 10.0, 10.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        ],
        dtype=torch.get_default_dtype(),
    )
    num_atoms = torch.tensor([2, 1])
    neighborhood = get_neighborhood_for_batch(
        frac_coords, lattice, num_atoms=num_atoms, cutoff=20
    )
    print(neighborhood)


def test_shift_lattice():
    lattice = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.get_default_dtype(),
    )
    supercells = torch.tensor([[0, 0, 0], [1, 1, 1]])
    frac_coords = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.5, 0.5, 0.5]],
        dtype=torch.get_default_dtype(),
    )
    num_atoms = torch.tensor([2, 1])
    cart_coords = atom_cart_coords_in_supercell(
        lattice, supercells, num_atoms, frac_coords
    )
    print(cart_coords)


if __name__ == "__main__":
    test_get_neighborhood()
