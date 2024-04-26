import itertools
from typing import Optional, Tuple

import numpy as np
from matscipy.neighbours import neighbour_list
import torch
import torch_geometric


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


# converts the coords into carteisan, and tiles them into the supercells
# you receive a [num_supercells * num_atoms, 3] tensor
# we use repeat_interleave, so the first <num_supercells> coords are the same as the first atom
def atom_cart_coords_in_supercell(
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
    shifted_adjustment = (shifts * lattices_for_cells).sum(dim=1)
    return cart_coords + shifted_adjustment, shifts


def get_neighborhood_for_batch(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    num_atoms: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    supercells = SUPERCELLS
    # the same as the above, but we always assume periodic boundary conditions AND that each input has a batch dimension
    supercell_cart_coords, shifts = atom_cart_coords_in_supercell(
        lattice,
        supercells,
        num_atoms,
        frac_coords,
    )
    edge_index = get_edge_index_for_center_cells(
        supercells, lattice, supercell_cart_coords, cutoff, num_atoms
    )
    # cart_coords = frac_to_cart_coords(frac_coords, lattice, num_atoms)

    # we need to repeat all the cartesian coords so we can compare the coords in the center cell with each of the atoms in the supercells
    # cart_coords = torch.repeat_interleave(cart_coords, num_supercells, dim=0)
    # distances = get_distances(cart_coords, supercell_cart_coords)
    # indices = get_indices_within_cutoff(distances, cutoff)

    return edge_index


def get_edge_index_for_center_cells(
    supercells,
    lattice,
    cart_coords,
    cutoff,
    num_atoms,
):
    batch_indexes = torch.arange(lattice.shape[0]).repeat_interleave(
        num_atoms * supercells.shape[0], dim=0
    )
    is_self_loops_in_result = False
    all_edge_index = torch_geometric.nn.radius_graph(
        cart_coords,
        cutoff,
        batch=batch_indexes,
        loop=is_self_loops_in_result,
        max_num_neighbors=32,
        flow="source_to_target",
        num_workers=1,
    )
    # now return all the indices where the src node is in the first dimension of the result
    # I used this to verify that the edge_index points to the nodes in its own graph. Make sure they're both the same!:
    # batch_indexes[all_edge_index[1]][1200:1800]
    # batch_indexes[all_edge_index[0]][1200:1800]
    return all_edge_index


def get_indices_within_cutoff(distances, cutoff):
    # Create a mask for distances less than cutoff and greater than 0
    mask = (distances < cutoff) & (distances > 0)

    # Get the indices where the mask is True
    indices = torch.nonzero(mask, as_tuple=True)

    # Extract the i and j indices
    i_indices = indices[0]
    j_indices = indices[1]

    # Combine i_indices and j_indices into a 2xN tensor
    return torch.stack((i_indices, j_indices), dim=0)


def get_distances(tensor1, tensor2):
    # Ensure the tensors have the same shape
    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape"

    # Calculate the squared differences between corresponding coordinates
    squared_diffs = (tensor1 - tensor2) ** 2

    # Sum the squared differences along the last dimension (coordinate dimension)
    squared_distances = torch.sum(squared_diffs, dim=-1)

    # Take the square root to get the Euclidean distances
    distances = torch.sqrt(squared_distances)

    return distances
