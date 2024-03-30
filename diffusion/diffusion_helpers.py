import torch
import torch.nn as nn
import numpy as np
import itertools
from torch_scatter import scatter

SUPERCELLS = torch.FloatTensor(list(itertools.product((-1, 0, 1), repeat=3)))
EPSILON = 1e-8


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x * self.W[None, :] * 2 * np.pi
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
            (adjacent_sigmas**2 * (sigmas**2 - adjacent_sigmas**2))
            / (sigmas**2)
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

    def reverse(self, ht, eps_h, t):
        alpha = 1 - self.betas[t]
        alpha = alpha.clamp_min(1 - self.betas[-2])
        alpha_bar = self.alpha_bars[t]
        sigma = self.sigmas[t].view(-1, 1)

        z = torch.where(
            (t > 1)[:, None].expand_as(ht),
            torch.randn_like(ht),
            torch.zeros_like(ht),
        )

        return (1.0 / torch.sqrt(alpha + EPSILON)).view(-1, 1) * (
            ht - ((1 - alpha) / torch.sqrt(1 - alpha_bar + EPSILON)).view(-1, 1) * eps_h
        ) + sigma * z

def frac_to_cart_coords(
    frac_coords,
    lattice,
    num_atoms,
):
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

def subtract_cog(x, num_atoms):
    batch = torch.arange(num_atoms.size(0), device=num_atoms.device).repeat_interleave(
        num_atoms, dim=0
    )
    cog = scatter(x, batch, dim=0, reduce="mean").repeat_interleave(num_atoms, dim=0)
    return x - cog