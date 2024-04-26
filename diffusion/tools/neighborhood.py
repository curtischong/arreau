import itertools
from typing import Optional, Tuple

import numpy as np
from matscipy.neighbours import neighbour_list
import torch

from diffusion.diffusion_helpers import frac_to_cart_coords


SUPERCELLS = torch.DoubleTensor(list(itertools.product((-1, 0, 1), repeat=3)))


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False,
) -> Tuple[np.ndarray, np.ndarray]:
    if pbc is None:
        pbc = (False, False, False)

    if cell is None or cell.any() == np.zeros((3, 3)).any():
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)

    pbc_x = pbc[0]
    pbc_y = pbc[1]
    pbc_z = pbc[2]
    identity = np.identity(3, dtype=float)
    max_positions = np.max(np.absolute(positions)) + 1
    # Extend cell in non-periodic directions
    # For models with more than 5 layers, the multiplicative constant needs to be increased.
    if not pbc_x:
        cell[:, 0] = max_positions * 5 * cutoff * identity[:, 0]
    if not pbc_y:
        cell[:, 1] = max_positions * 5 * cutoff * identity[:, 1]
    if not pbc_z:
        cell[:, 2] = max_positions * 5 * cutoff * identity[:, 2]

    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
        # self_interaction=True,  # we want edges from atom to itself in different periodic images
        # use_scaled_positions=False,  # positions are not scaled positions
    )

    if not true_self_interaction:
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = sender == receiver
        true_self_edge &= np.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    # Build output
    edge_index = np.stack((sender, receiver))  # [2, n_edges]

    # From the docs: With the shift vector S, the distances D between atoms can be computed from
    # D = positions[j]-positions[i]+S.dot(cell)
    shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]

    return edge_index, shifts, unit_shifts


def shift_lattice(
    lattices: torch.Tensor,  # [batch, 3, 3]
    supercells: torch.Tensor,
    num_atoms: torch.Tensor,
    frac_coords: torch.Tensor,
):
    shifts = supercells.repeat(num_atoms.sum(), 1).unsqueeze(1)
    num_supercells = supercells.shape[0]

    adjusted_num_atoms = (
        num_supercells * num_atoms
    )  # since we need to put the atom in each supercell
    lattices_for_cells = torch.repeat_interleave(lattices, adjusted_num_atoms, dim=0)
    frac_coords_for_cells = torch.repeat_interleave(frac_coords, num_supercells, dim=0)
    cart_coords = torch.einsum(
        "bi,bij->bj", frac_coords_for_cells, lattices_for_cells
    )  # cart coords
    return cart_coords + (shifts * lattices_for_cells)


def get_neighborhood_for_batch(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    num_atoms: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    # the same as the above, but we always assume periodic boundary conditions AND that each input has a batch dimension
    batch_size = num_atoms.shape[0]

    # Get the positions for each atom
    cart_coords = frac_to_cart_coords(frac_coords, lattice, num_atoms)

    # unit_cell = torch.tensor(
    #     SUPERCELLS, device=lattice.device, dtype=torch.get_default_dtype()
    # )
    # num_cells = len(unit_cell)
    # unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(cart_coords2), 1, 1)
    # unit_cell = torch.transpose(unit_cell, 0, 1)
    # unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Compute the x, y, z positional offsets for each cell in each image
    # data_cell = torch.transpose(lattice, 1, 2)
    lattice_translations_batch = SUPERCELLS.repeat(batch_size, 1).view(
        batch_size, 27, 3
    )
    # pbc_offsets = torch.bmm(lattice_translations_batch, lattice)
    # pbc_offsets = lattice_translations_batch, lattice)
    # TODO: verify unit_cell_batch is correct
    lattice_translations_batch = lattice_translations_batch.unsqueeze(-1)
    lattice = lattice.unsqueeze(1)

    result = lattice_translations_batch * lattice
    result = result.view(-1, 3, 3)  # [batch_size * 27, 3, 3]
    # result has shape (batch_size, 27, 3, 3)

    coords = frac_coords.repeat(27, 1)

    pbc_offsets_per_atom = torch.repeat_interleave(
        lattice_translations_batch, num_atoms, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = cart_coords.view(-1, 3, 1).expand(-1, -1, num_cells)
    all_cell_coords = pos1 + pbc_offsets_per_atom
    # Now for each atom, get the closest neighbor
    # pos2 = cart_coords.view(-1, 3, 1).expand(-1, -1, num_cells)
    # Add the PBC offsets for the second atom
    # curtis: why not for the first coords too? Cause the first coord is the center cell. We just expand it to have dim 27 so we can calculate the difference between pos2
    res = get_nearest_neighbors(cart_coords, all_cell_coords, cutoff)
    return res

    # # Compute the vector between atoms
    # # shape (num_atom_squared_sum, 3, 27)
    # atom_distance_vector = pos1 - pos2
    # atom_distance_sqr = torch.sum(
    #     atom_distance_vector**2, dim=1
    # )  # There is an optimization here. we take the distance^2 (rather than distance) since distance is monotonically increasing

    # min_atom_distance_sqr, min_indices = atom_distance_sqr.min(dim=-1)

    # return_list = [min_atom_distance_sqr]

    # if return_vector:
    #     min_indices = min_indices[:, None, None].repeat([1, 3, 1])

    #     min_atom_distance_vector = torch.gather(
    #         atom_distance_vector, 2, min_indices
    #     ).squeeze(-1)

    #     return_list.append(min_atom_distance_vector)

    # if return_to_jimages:
    #     to_jimages = unit_cell.T[min_indices].long()
    #     return_list.append(to_jimages)

    # return return_list[0] if len(return_list) == 1 else return_list


def get_nearest_neighbors(
    initial_coords, coords: torch.Tensor, cutoff: float
) -> torch.Tensor:
    # Calculate pairwise distances between coordinates
    distances = torch.cdist(initial_coords, coords)

    # Find the indices of the k nearest neighbors (excluding the i-th coordinate itself)
    # _, indices = torch.topk(distances.squeeze(), k + 1, largest=False)
    # indices = indices[1:]  # Exclude the i-th coordinate itself
    # Find the indices of the coordinates within radius r (excluding the i-th coordinate itself)
    indices = torch.where((distances <= cutoff) & (distances > 0))
    return indices
