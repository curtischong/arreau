from dataclasses import dataclass
from typing import Optional
import torch

from torch_scatter import scatter
from torch_geometric.data import Batch
import torchmetrics
from tqdm import tqdm
import numpy as np

from diffusion.diffusion_helpers import (
    VP,
    VE_pbc,
    VP_lattice,
    frac_to_cart_coords,
    polar_decomposition,
    radius_graph_pbc,
    subtract_cog,
    symmetric_matrix_to_vector,
    vector_to_symmetric_matrix,
)
from diffusion.tools.atomic_number_table import (
    AtomicNumberTable,
    atomic_symbols_to_indices,
)
from diffusion.inference.visualize_crystal import (
    VisualizationSetting,
    vis_crystal_during_sampling,
)
from torch.nn import functional as F


pos_sigma_min = 0.001
pos_sigma_max = 10.0

type_power = 2
lattice_power = 2
type_clipmax = 0.999
lattice_clipmax = 0.999


@dataclass
class SampleResult:
    frac_x: Optional[np.ndarray] = None
    atomic_numbers: Optional[np.ndarray] = None

    # crystal-wide information. If there are m crystals, these arrays have m indexes
    lattice: Optional[np.ndarray] = None
    idx_start: Optional[np.ndarray] = (
        None  # The index (in x, h) of the first atom in each crystal
    )
    num_atoms: Optional[np.ndarray] = None  # The number of atoms in each crystal


class DiffusionLossMetric(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def compute(self):
        return self.total_loss / self.total_samples

    def update(self, loss, batch: Batch):
        self.total_loss += loss.sum()
        num_batches = torch.unique(batch.batch).size(0)
        self.total_samples += num_batches


class DiffusionLoss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cutoff = args.radius
        self.max_neighbors = args.max_neighbors
        self.T = args.num_timesteps
        self.pos_diffusion = VE_pbc(
            self.T, sigma_min=pos_sigma_min, sigma_max=pos_sigma_max
        )

        self.type_diffusion = VP(
            num_steps=self.T,
            power=type_power,
            clipmax=type_clipmax,
        )

        self.lattice_diffusion = VP_lattice(
            num_steps=self.T,
            power=lattice_power,
            clipmax=lattice_clipmax,
        )

        self.cost_coord_coeff = 1
        self.cost_type_coeff = 1
        self.lattice_coeff = 1
        # self.norm_x = 10. # I'm not sure why mofdiff uses these values. I'm going to use 1.0 for now.
        # self.norm_h = 10.
        self.norm_x = 1.0
        self.norm_h = 1.0

    def compute_error_for_global_vec(self, pred_eps, eps, weights=None):
        """Computes error, i.e. the most likely prediction of x."""
        error = (eps - pred_eps) ** 2
        if weights is not None:
            error *= weights
        if len(error.shape) > 1:
            error = error.sum()
        return error

    def compute_error(self, pred_eps, eps, batch, weights=None):
        """Computes error, i.e. the most likely prediction of x."""
        if weights is None:
            error = scatter(((eps - pred_eps) ** 2), batch.batch, dim=0, reduce="mean")
        else:
            error = scatter(
                weights * ((eps - pred_eps) ** 2), batch.batch, dim=0, reduce="mean"
            )
        if len(error.shape) > 1:
            error = error.sum(-1)
        return error

    def phi(
        self,
        frac_x_t: torch.Tensor,
        h_t: torch.Tensor,
        t_int: torch.Tensor,
        num_atoms: torch.Tensor,
        lattice: torch.Tensor,
        noisy_symmetric_vector: torch.Tensor,
        model,
        batch: Batch,
        t_emb_weights,
    ):
        t = self.type_diffusion.betas[t_int].view(-1, 1)
        t_emb = t_emb_weights(t)

        noisy_symmetric_vector = torch.repeat_interleave(
            noisy_symmetric_vector, num_atoms, dim=0
        )

        scalar_feats = torch.cat([h_t, t_emb, noisy_symmetric_vector], dim=1)
        cart_x_t = frac_to_cart_coords(frac_x_t, lattice, num_atoms)

        lattice_feat = torch.repeat_interleave(lattice, num_atoms, dim=0)

        # overwrite the batch with the new values. I'm not making a new batch object since I may miss some attributes.
        # If overwritting leads to problems, we'll need to make a new Batch object
        batch.x = scalar_feats
        batch.pos = cart_x_t
        batch.vec = torch.cat([frac_x_t.unsqueeze(1), lattice_feat], dim=1)

        # we need to overwrite the edge_index for the batch since when we add noise to the positions, some atoms may be
        # so far apart from each other they are no longer considered neighbors. So we need to recompute the neighbors.
        edge_index, cell_offsets, neighbors = radius_graph_pbc(
            cart_x_t,
            lattice,
            batch.num_atoms,
            self.cutoff,
            self.max_neighbors,
            device=cart_x_t.device,
        )
        batch.edge_index = edge_index

        # compute the predictions
        pred_eps_h, pred_eps_x, pred_symmetric_vector_noise, _pred_eps_global_vec = (
            model(batch)
        )

        # normalize the predictions
        used_sigmas_x = self.pos_diffusion.sigmas[t_int].view(-1, 1)
        pred_eps_x = subtract_cog(pred_eps_x, num_atoms)

        return (
            pred_eps_x.squeeze(1) / used_sigmas_x,
            pred_eps_h,
            pred_symmetric_vector_noise,
        )

    def normalize(self, x, h):
        x = x / self.norm_x
        h = h / self.norm_h
        return x, h

    def unnormalize(self, x, h):
        x = x * self.norm_x
        h = h * self.norm_h
        return x, h

    def diffuse_lattice_params(self, lattice: torch.Tensor, t_int: torch.Tensor):
        # the diffusion happens on the symmetric positive-definite matrix part, but we will pass in vectors and receive vectors out from the model.
        # This is so the model can use vector features for the equivariance

        rotation_matrix, symmetric_matrix = polar_decomposition(lattice)
        symmetric_matrix_vector = symmetric_matrix_to_vector(symmetric_matrix)

        noisy_symmetric_vector, noise_vector = self.lattice_diffusion(
            symmetric_matrix_vector, t_int
        )
        noisy_symmetric_matrix = vector_to_symmetric_matrix(noisy_symmetric_vector)
        noisy_lattice = rotation_matrix @ noisy_symmetric_matrix

        # given the noisy_symmetric_vector, it needs to predict the noise vector
        # when sampling, we get take hte predicted noise vector to get the unnoised symmetric vecotr, which we can convert into a symmetric matrix, which is the lattice
        return (
            noisy_lattice,
            noisy_symmetric_vector,
            noise_vector,
            symmetric_matrix_vector,
        )

    def __call__(self, model, batch, t_emb_weights, t_int=None):
        cart_x_0 = batch.pos.squeeze(0)
        h = batch.A0
        lattice = batch.L0
        lattice = lattice.view(-1, 3, 3)
        num_atoms = batch.num_atoms

        cart_x_0, h = self.normalize(cart_x_0, h)

        # Sample a timestep t.
        # TODO: can we simplify this? is t_int always None? Verification code may inconsistently pass in t_int vs train code
        if t_int is None:
            t_int = torch.randint(
                1, self.T + 1, size=(num_atoms.size(0), 1), device=cart_x_0.device
            ).long()
        else:
            t_int = (
                torch.ones((batch.num_atoms.size(0), 1), device=cart_x_0.device).long()
                * t_int
            )
        t_int_atoms = t_int.repeat_interleave(num_atoms, dim=0)

        # Sample noise.
        frac_x_t, target_eps_x, used_sigmas_x = self.pos_diffusion(
            cart_x_0, t_int_atoms, lattice, num_atoms
        )
        h_t, eps_h = self.type_diffusion(h, t_int_atoms)  # eps is the noise
        (
            noisy_lattice,
            noisy_symmetric_vector,
            _symmetric_vector_noise,
            symmetric_matrix_vector,
        ) = self.diffuse_lattice_params(lattice, t_int)

        # Compute the prediction.
        pred_eps_x, pred_eps_h, pred_symmetric_vector = self.phi(
            frac_x_t,
            h_t,
            t_int_atoms,
            num_atoms,
            noisy_lattice,
            noisy_symmetric_vector,
            model,
            batch,
            t_emb_weights,
        )

        # Compute the error.
        error_x = self.compute_error(
            pred_eps_x,
            target_eps_x / used_sigmas_x**2,
            batch,
            0.5 * used_sigmas_x**2,
        )  # likelihood reweighting
        error_h = self.compute_error(pred_eps_h, eps_h, batch)
        error_l = F.mse_loss(pred_symmetric_vector, symmetric_matrix_vector)

        loss = (
            self.cost_coord_coeff * error_x
            + self.cost_type_coeff * error_h
            + self.lattice_coeff * error_l
        )
        return loss.mean()

    @torch.no_grad()
    def sample(
        self,
        *,
        model,
        z_table: AtomicNumberTable,
        t_emb_weights,  # TODO: can we store this in diffusion_loss? rather than passing it in every time?
        num_atoms_per_sample: int,
        num_samples_in_batch: int,
        vis_name: str,
        visualization_setting: VisualizationSetting,
        show_bonds: bool,
    ) -> SampleResult:
        num_atomic_states = len(z_table)

        lattice = torch.randn([num_samples_in_batch, 3, 3])

        # TODO: verify that we are uing the GPU during inferencing (via nvidia smi)
        # I am not 100% sure that pytorch lightning is using the GPU during inferencing.
        frac_x = (
            torch.randn(
                [num_samples_in_batch * num_atoms_per_sample, 3],
                dtype=torch.get_default_dtype(),
            )
            * pos_sigma_max
        )
        num_atoms = torch.full((num_samples_in_batch,), num_atoms_per_sample)

        h = torch.randn([num_atoms.sum(), num_atomic_states])
        constant_atoms = atomic_symbols_to_indices(
            z_table,
            ["C", "C", "C", "C", "C", "C", "C", "C"],
            # ["Cl", "Cl", "Cl", "Cl", "Na", "Na", "Na", "Na"],
            # ["Na", "Na", "Na", "Na", "Cl"],
        )

        for timestep in tqdm(reversed(range(1, self.T))):
            t = torch.full((num_atoms.sum(),), fill_value=timestep)
            timestep_vec = torch.tensor([timestep])  # add a batch dimension

            rotation_matrix, symmetric_matrix = polar_decomposition(lattice)
            symmetric_vector = symmetric_matrix_to_vector(symmetric_matrix)

            h = constant_atoms
            score_x, score_h, pred_symmetric_vector = self.phi(
                frac_x,
                h,
                t,
                num_atoms,
                lattice,
                symmetric_vector,
                model,
                Batch(
                    num_atoms=num_atoms, batch=torch.tensor(0).repeat(num_atoms.sum())
                ),
                t_emb_weights,
            )
            next_symmetric_vector = self.lattice_diffusion.reverse(
                symmetric_vector, pred_symmetric_vector, timestep_vec
            )

            next_symmetric_matrix = vector_to_symmetric_matrix(next_symmetric_vector)
            lattice = rotation_matrix @ next_symmetric_matrix

            cart_x = frac_to_cart_coords(frac_x, lattice, num_atoms)
            frac_x = self.pos_diffusion.reverse(cart_x, score_x, t, lattice, num_atoms)
            h = self.type_diffusion.reverse(h, score_h, t)
            h = constant_atoms

            if (
                visualization_setting == VisualizationSetting.ALL
                and (timestep != self.T - 1)
                and (timestep % 10 == 0)
            ):
                vis_crystal_during_sampling(
                    z_table, h, lattice, frac_x, vis_name + f"_{timestep}", show_bonds
                )

        frac_x, h = self.unnormalize(
            frac_x, h
        )  # why does mofdiff unnormalize? The fractional x coords can be > 1 after unormalizing.

        if visualization_setting != VisualizationSetting.NONE:
            vis_crystal_during_sampling(
                z_table, h, lattice, frac_x, vis_name + "_final", show_bonds
            )
        h_best_idx = torch.argmax(h, dim=1).numpy()
        atomic_numbers = np.vectorize(z_table.index_to_z)(h_best_idx)
        return SampleResult(
            num_atoms=num_atoms.numpy(),
            frac_x=frac_x.numpy(),
            atomic_numbers=atomic_numbers,
            lattice=lattice.numpy(),
        )
