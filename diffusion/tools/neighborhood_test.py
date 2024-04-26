from diffusion.tools.neighborhood import get_neighborhood_for_batch, shift_lattice
import torch


def test_get_neighborhood():
    # frac_coords = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [0.5, 0.5, 0.5]])
    torch.set_default_dtype(torch.float64)
    frac_coords = torch.tensor(
        [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.8, 0.8]],
        dtype=torch.get_default_dtype(),
    )
    lattice = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[10.0, 10.0, 10.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        ],
        dtype=torch.get_default_dtype(),
    )
    num_atoms = torch.tensor([1, 2])
    neighborhood = get_neighborhood_for_batch(
        frac_coords, lattice, num_atoms=num_atoms, cutoff=1
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
    shifted_lattices = shift_lattice(lattice, supercells, num_atoms, frac_coords)
    print(shifted_lattices)


if __name__ == "__main__":
    test_shift_lattice()
