import torch
import torch.nn as nn
import numpy as np
import itertools
from torch_scatter import scatter
import copy

SUPERCELLS = torch.DoubleTensor(list(itertools.product((-1, 0, 1), repeat=3)))
EPSILON = 1e-8


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.gaussian_fourier_proj_w = nn.Parameter(
            torch.randn(embedding_size) * scale, requires_grad=False
        )

    def forward(self, x):
        x_proj = x * self.gaussian_fourier_proj_w[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class VE_pbc(nn.Module):
    """
    variance exploding diffusion under periodic boundary condition.
    """

    def __init__(self, num_steps, sigma_min, sigma_max):
        super().__init__()
        self.T = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.register_buffer(
            "sigmas",
            torch.exp(torch.linspace(np.log(sigma_min), np.log(sigma_max), self.T + 1)),
        )

    def forward(self, x0, t, lattice, num_atoms, **kwargs):
        """
        x0 should be wrapped cart coords.
        """
        used_sigmas = self.sigmas[t].view(-1, 1)
        eps_x = torch.randn_like(x0) * used_sigmas
        frac_p_noisy = cart_to_frac_coords(x0 + eps_x, lattice, num_atoms)
        cart_p_noisy = frac_to_cart_coords(frac_p_noisy, lattice, num_atoms)
        _, wrapped_eps_x = min_distance_sqr_pbc(
            cart_p_noisy,
            x0,
            lattice,
            num_atoms,
            x0.device,
            return_vector=True,
        )
        # wrapped_eps_x is like the noise. it's a vector tat points from the noisy coords to the clean x0 coord.
        return frac_p_noisy, wrapped_eps_x, used_sigmas

    def reverse(self, xt, epx_x, t, lattice, num_atoms):
        """
        xt should be wrapped cart coords.
        """
        sigmas = self.sigmas[t].view(-1, 1)
        adjacent_sigmas = torch.where(
            (t == 0).view(-1, 1),
            torch.zeros_like(sigmas),
            self.sigmas[t - 1].view(-1, 1),
        )
        cart_p_mean = xt - epx_x * (sigmas**2 - adjacent_sigmas**2)
        # the sign of eps_p here is related to the verification above.
        cart_p_rand = torch.sqrt(
            (adjacent_sigmas**2 * (sigmas**2 - adjacent_sigmas**2)) / (sigmas**2)
        ) * torch.randn_like(xt)
        cart_p_next = cart_p_mean + cart_p_rand  # before wrapping
        frac_p_next = cart_to_frac_coords(cart_p_next, lattice, num_atoms)
        return frac_p_next


class VP(nn.Module):
    """
    variance preserving diffusion.
    """

    def __init__(self, num_steps=1000, s=0.0001, power=2, clipmax=0.999):
        super().__init__()
        t = torch.arange(0, num_steps + 1, dtype=torch.float)
        # play with the parameters here: https://www.desmos.com/calculator/jtb6whrvej
        # cosine schedule introduced in https://arxiv.org/abs/2102.09672
        f_t = torch.cos((np.pi / 2) * ((t / num_steps) + s) / (1 + s)) ** power
        alpha_bars = f_t / f_t[0]
        betas = torch.cat(
            [torch.zeros([1]), 1 - (alpha_bars[1:] / alpha_bars[:-1])], dim=0
        )
        betas = betas.clamp_max(clipmax)
        sigmas = torch.sqrt(betas[1:] * ((1 - alpha_bars[:-1]) / (1 - alpha_bars[1:])))
        sigmas = torch.cat([torch.zeros([1]), sigmas], dim=0)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("betas", betas)
        self.register_buffer("sigmas", sigmas)

    def forward(self, h0, t):
        alpha_bar = self.alpha_bars[t]
        eps = torch.randn_like(h0)
        ht = (
            torch.sqrt(alpha_bar).view(-1, 1) * h0
            + torch.sqrt(1 - alpha_bar).view(-1, 1) * eps
        )
        return ht, eps

    # This formula is from algorithm 2 sampling from https://arxiv.org/pdf/2006.11239.pdf
    def reverse(self, ht, h0, t):
        eps_h = ht - h0  # this is the predicted noise since we're predicting h0
        alpha = 1 - self.betas[t]
        alpha = alpha.clamp_min(1 - self.betas[-2])
        alpha_bar = self.alpha_bars[t]
        sigma = self.sigmas[t].view(-1, 1)

        # This is noise we add so when we do the backwards sample, we don't collapse to one point
        z = torch.where(
            (t > 1)[:, None].expand_as(ht),
            torch.randn_like(ht),
            torch.zeros_like(ht),
        )

        return (1.0 / torch.sqrt(alpha + EPSILON)).view(-1, 1) * (
            ht - ((1 - alpha) / torch.sqrt(1 - alpha_bar + EPSILON)).view(-1, 1) * eps_h
        ) + sigma * z


class VP_lattice(nn.Module):
    """
    variance preserving diffusion.
    """

    def __init__(self, num_steps=1000, s=0.0001, power=2, clipmax=0.999):
        super().__init__()
        t = torch.arange(0, num_steps + 1, dtype=torch.float)
        # play with the parameters here: https://www.desmos.com/calculator/jtb6whrvej
        # cosine schedule introduced in https://arxiv.org/abs/2102.09672
        f_t = torch.cos((np.pi / 2) * ((t / num_steps) + s) / (1 + s)) ** power
        alpha_bars = f_t / f_t[0]
        betas = torch.cat(
            [torch.zeros([1]), 1 - (alpha_bars[1:] / alpha_bars[:-1])], dim=0
        )
        betas = betas.clamp_max(clipmax)
        sigmas = torch.sqrt(betas[1:] * ((1 - alpha_bars[:-1]) / (1 - alpha_bars[1:])))
        sigmas = torch.cat([torch.zeros([1]), sigmas], dim=0)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("betas", betas)
        self.register_buffer("sigmas", sigmas)

    def forward(self, h0, t):
        alpha_bar = self.alpha_bars[t]
        eps = torch.randn_like(h0)
        ht = (
            torch.sqrt(alpha_bar).view(-1, 1) * h0
            + torch.sqrt(1 - alpha_bar).view(-1, 1) * eps
        )
        return ht, eps

    # since the model predicts l0, the reverse function is different from the normal VP diffusion.
    # we are "mixing" the predicted l0 and the current lt to get lt-1
    def reverse(self, lt, predicted_l0, pred_lattice, t):
        alpha = 1 - self.betas[t]
        alpha = alpha.clamp_min(1 - self.betas[-2])
        alpha_bar = self.alpha_bars[t]
        sigma = self.sigmas[t].view(-1, 1)

        _, pred_lattice_symmetric_matrix = polar_decomposition(pred_lattice)
        pred_lattice_symmetric_vector = symmetric_matrix_to_vector(
            pred_lattice_symmetric_matrix
        )

        # predicted_noise = lt - predicted_l0
        predicted_noise = lt - ((pred_lattice_symmetric_vector + predicted_l0) / 2)

        # This is noise we add so when we do the backwards sample, we don't collapse to one point
        z = torch.where(
            (t > 1)[:, None].expand_as(lt),
            torch.randn_like(lt),
            torch.zeros_like(lt),
        )

        return (1.0 / torch.sqrt(alpha + EPSILON)).view(-1, 1) * (
            lt
            - ((1 - alpha) / torch.sqrt(1 - alpha_bar + EPSILON)).view(-1, 1)
            * predicted_noise
        ) + sigma * z

    # def normalizing_mean_constant(self, n: torch.Tensor):
    #     avg_density_of_dataset = 0.05539856385043283
    #     c = 1 / avg_density_of_dataset
    #     return torch.pow(n * c, 1 / 3)

    # def normalizing_variance_constant(self, n: torch.Tensor):
    #     v = 152.51649752530176  # assuming that v is the average volume of the dataset
    #     v = v / 6  # This is an adjustment I think will lead to more stable volumes
    #     return torch.pow(n * v, 1 / 3)


def frac_to_cart_coords(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    num_atoms: torch.Tensor,
) -> torch.Tensor:
    lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
    pos = torch.einsum("bi,bij->bj", frac_coords, lattice_nodes)  # cart coords
    return pos


def cart_to_frac_coords(
    cart_coords,
    lattice,
    num_atoms,
):
    # use pinv in case the predicted lattice is not rank 3
    inv_lattice = torch.linalg.pinv(lattice)
    inv_lattice_nodes = torch.repeat_interleave(inv_lattice, num_atoms, dim=0)
    frac_coords = torch.einsum("bi,bij->bj", cart_coords, inv_lattice_nodes)
    return frac_coords % 1.0


def min_distance_sqr_pbc(
    cart_coords1,
    cart_coords2,
    lattice,
    num_atoms,
    device,
    return_vector=False,
    return_to_jimages=False,
):
    """Compute the pbc distance between atoms in cart_coords1 and cart_coords2.
    This function assumes that cart_coords1 and cart_coords2 have the same number of atoms
    in each data point.
    returns:
        basic return:
            min_atom_distance_sqr: (N_atoms, )
        return_vector == True:
            min_atom_distance_vector: vector pointing from cart_coords1 to cart_coords2, (N_atoms, 3)
        return_to_jimages == True:
            to_jimages: (N_atoms, 3), position of cart_coord2 relative to cart_coord1 in pbc
    """
    batch_size = len(num_atoms)

    # Get the positions for each atom
    pos1 = cart_coords1
    pos2 = cart_coords2

    unit_cell = torch.tensor(SUPERCELLS, device=device, dtype=torch.get_default_dtype())
    num_cells = len(unit_cell)
    # unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(cart_coords2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(lattice, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(pbc_offsets, num_atoms, dim=0)

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the vector between atoms
    # shape (num_atom_squared_sum, 3, 27)
    atom_distance_vector = pos1 - pos2
    atom_distance_sqr = torch.sum(atom_distance_vector**2, dim=1)

    min_atom_distance_sqr, min_indices = atom_distance_sqr.min(dim=-1)

    return_list = [min_atom_distance_sqr]

    if return_vector:
        min_indices = min_indices[:, None, None].repeat([1, 3, 1])

        min_atom_distance_vector = torch.gather(
            atom_distance_vector, 2, min_indices
        ).squeeze(-1)

        return_list.append(min_atom_distance_vector)

    if return_to_jimages:
        to_jimages = unit_cell.T[min_indices].long()
        return_list.append(to_jimages)

    return return_list[0] if len(return_list) == 1 else return_list


# cog = center of gravity (it centers each crystal)
def subtract_cog(x, num_atoms):
    batch = torch.arange(num_atoms.size(0), device=num_atoms.device).repeat_interleave(
        num_atoms, dim=0
    )
    cog = scatter(x, batch, dim=0, reduce="mean").repeat_interleave(num_atoms, dim=0)
    return x - cog


def radius_graph_pbc(
    cart_coords,
    lattice,
    num_atoms,
    radius,
    max_num_neighbors_threshold,
    device,
    topk_per_pair=None,
    remove_self_edges=True,
):
    """Computes pbc graph edges under pbc.

    topk_per_pair: (num_atom_pairs,), select topk edges per atom pair

    Note: topk should take into account self-self edge for (i, i)
    """
    batch_size = len(num_atoms)

    # position of the atoms
    atom_pos = cart_coords

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = num_atoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # index offset between images
    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image

    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="trunc")
    ).long() + index_offset_expand
    index2 = (atom_count_sqr % num_atoms_per_image_expand).long() + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    unit_cell = torch.tensor(SUPERCELLS, device=device, dtype=torch.get_default_dtype())
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(lattice, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)

    if topk_per_pair is not None:
        assert topk_per_pair.size(0) == num_atom_pairs
        atom_distance_sqr_sort_index = torch.argsort(atom_distance_sqr, dim=1)
        assert atom_distance_sqr_sort_index.size() == (num_atom_pairs, num_cells)
        atom_distance_sqr_sort_index = (
            atom_distance_sqr_sort_index
            + torch.arange(num_atom_pairs, device=device)[:, None] * num_cells
        ).view(-1)
        topk_mask = (
            torch.arange(num_cells, device=device)[None, :] < topk_per_pair[:, None]
        )
        topk_mask = topk_mask.view(-1)
        topk_indices = atom_distance_sqr_sort_index.masked_select(topk_mask)

        topk_mask = torch.zeros(num_atom_pairs * num_cells, device=device)
        topk_mask.scatter_(0, topk_indices, 1.0)
        topk_mask = topk_mask.bool()

    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    if remove_self_edges:
        mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
        mask = torch.logical_and(mask_within_radius, mask_not_same)
    else:
        mask = mask_within_radius
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    if topk_per_pair is not None:
        topk_mask = torch.masked_select(topk_mask, mask)

    num_neighbors = torch.zeros(len(cart_coords), device=device)
    num_neighbors.index_add_(0, index1, torch.ones(len(index1), device=device))
    num_neighbors = num_neighbors.long()
    max_num_neighbors = torch.max(num_neighbors).long()

    # Compute neighbors per image
    _max_neighbors = copy.deepcopy(num_neighbors)
    _max_neighbors[_max_neighbors > max_num_neighbors_threshold] = (
        max_num_neighbors_threshold
    )
    _num_neighbors = torch.zeros(len(cart_coords) + 1, device=device).long()
    _natoms = torch.zeros(num_atoms.shape[0] + 1, device=device).long()
    _num_neighbors[1:] = torch.cumsum(_max_neighbors, dim=0)
    _natoms[1:] = torch.cumsum(num_atoms, dim=0)
    num_neighbors_image = _num_neighbors[_natoms[1:]] - _num_neighbors[_natoms[:-1]]

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        if topk_per_pair is None:
            return torch.stack((index2, index1)), -unit_cell, num_neighbors_image
        else:
            return (
                torch.stack((index2, index1)),
                -unit_cell,
                num_neighbors_image,
                topk_mask,
            )

    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with values greater than radius*radius so we can easily remove unused distances later.
    distance_sort = torch.zeros(
        len(cart_coords) * max_num_neighbors, device=device
    ).fill_(radius * radius + 1.0)

    # Create an index map to map distances from atom_distance_sqr to distance_sort
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index1 * max_num_neighbors
        + torch.arange(len(index1), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance_sqr)
    distance_sort = distance_sort.view(len(cart_coords), max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]

    # Offset index_sort so that it indexes into index1
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors_threshold
    )
    # Remove "unused pairs" with distances greater than the radius
    mask_within_radius = torch.le(distance_sort, radius * radius)
    index_sort = torch.masked_select(index_sort, mask_within_radius)

    # At this point index_sort contains the index into index1 of the closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index1), device=device).bool()
    mask_num_neighbors.index_fill_(0, index_sort, True)

    # Finally mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
    index1 = torch.masked_select(index1, mask_num_neighbors)
    index2 = torch.masked_select(index2, mask_num_neighbors)
    unit_cell = torch.masked_select(
        unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)

    if topk_per_pair is not None:
        topk_mask = torch.masked_select(topk_mask, mask_num_neighbors)

    edge_index = torch.stack((index2, index1))

    # fix to to_jimages: negate unit_cell.
    if topk_per_pair is None:
        return edge_index, -unit_cell, num_neighbors_image
    else:
        return edge_index, -unit_cell, num_neighbors_image, topk_mask


def symmetrize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    return (matrix + matrix.transpose(1, 2)) / 2


# from claude and https://discuss.pytorch.org/t/polar-decomposition-of-matrices-in-pytorch/188458/2
def polar_decomposition(matrix: torch.Tensor):
    # Perform SVD on the input matrix
    U, S, Vt = torch.linalg.svd(matrix)
    u = U @ Vt

    # Compute the symmetric positive-definite matrix l_tilda
    L_tilda = torch.matmul(Vt.transpose(-2, -1), torch.matmul(torch.diag_embed(S), Vt))

    # enforce symmetry to avoid numerical instabilities
    L_tilda = symmetrize_matrix(L_tilda)

    return u, L_tilda


def symmetric_matrix_to_vector(matrix: torch.Tensor):
    """
    Converts a batch of 3x3 symmetric matrices to vectors containing the upper triangular part (including diagonal).
    """
    assert (
        matrix.dim() == 3
    ), "Input must be a batch of matrices with shape (batch_size, 3, 3)"
    assert matrix.shape[1:] == (3, 3), "Each matrix in the batch must be 3x3"

    assert torch.allclose(
        matrix, matrix.transpose(1, 2)
    ), "Each matrix in the batch must be symmetric"

    vector = torch.stack(
        [
            matrix[:, 0, 0],
            matrix[:, 0, 1],
            matrix[:, 0, 2],
            matrix[:, 1, 1],
            matrix[:, 1, 2],
            matrix[:, 2, 2],
        ],
        dim=1,
    )
    return vector


def vector_to_symmetric_matrix(vector: torch.Tensor):
    """
    Reconstructs a batch of 3x3 symmetric matrices from a batch of vectors containing the upper triangular part (including diagonal).
    """
    assert vector.shape[-1] == 6, "Last dimension of input vector must have 6 elements"
    batch_size = vector.shape[:-1]
    matrix = torch.zeros((*batch_size, 3, 3), dtype=vector.dtype, device=vector.device)
    matrix[:, 0, 0] = vector[:, 0]
    matrix[:, 0, 1] = matrix[:, 1, 0] = vector[:, 1]
    matrix[:, 0, 2] = matrix[:, 2, 0] = vector[:, 2]
    matrix[:, 1, 1] = vector[:, 3]
    matrix[:, 1, 2] = matrix[:, 2, 1] = vector[:, 4]
    matrix[:, 2, 2] = vector[:, 5]
    return matrix
