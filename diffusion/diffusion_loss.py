import torch

from torch_scatter import scatter
from torch_geometric.data import Batch
import torchmetrics
from tqdm import tqdm

from diffusion.diffusion_helpers import (
    VP,
    VE_pbc,
    cart_to_frac_coords,
    frac_to_cart_coords,
    polar_decomposition,
    radius_graph_pbc,
    subtract_cog,
    symmetric_matrix_to_vector,
    vector_to_symmetric_matrix,
)
from diffusion.tools.atomic_number_table import AtomicNumberTable
from diffusion.inference.visualize_crystal import vis_crystal_during_sampling


pos_sigma_min = 0.001
pos_sigma_max = 10.0

type_power = 2
lattice_power = 2
type_clipmax = 0.999
lattice_clipmax = 0.999


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

        self.lattice_diffusion = VP(
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
        x_t,
        h_t,
        t_int,
        num_atoms,
        lattice: torch.Tensor,
        model,
        batch: Batch,
        t_emb_weights,
        frac=False,
    ):
        t = self.type_diffusion.betas[t_int].view(-1, 1)
        t_emb = t_emb_weights(t)
        h_time = torch.cat([h_t, t_emb], dim=1)
        cart_x_t = x_t if not frac else frac_to_cart_coords(x_t, lattice, num_atoms)

        # overwrite the batch with the new values. I'm not making a new batch object since I may miss some attributes.
        # If overwritting leads to problems, we'll need to make a new Batch object
        batch.x = h_time
        batch.pos = cart_x_t
        batch.vec = torch.repeat_interleave(
            lattice, num_atoms, dim=0
        )  # This line is needed to have each node have it's corresponding lattice vector
        # perf. combine with frac_to_cart_coords above. since frac is always true, we're recomputing this twice

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
        pred_eps_h, pred_eps_x, pred_eps_l = model(batch)

        # normalize the predictions
        used_sigmas_x = self.pos_diffusion.sigmas[t_int].view(-1, 1)
        pred_eps_x = subtract_cog(pred_eps_x, num_atoms)
        return pred_eps_x.squeeze(1) / used_sigmas_x, pred_eps_h, pred_eps_l

    def normalize(self, x, h):
        x = x / self.norm_x
        h = h / self.norm_h
        return x, h

    def unnormalize(self, x, h):
        x = x * self.norm_x
        h = h * self.norm_h
        return x, h

    def diffuse_lattice(self, lattice, t_int):
        # the diffusion happens on the symmetric positive-definite matrix part, but we will pass in vectors and receive vectors out from the model.
        # This is so the model can use vector features for the equivariance

        rot_mat, symmetric_lattice = polar_decomposition(lattice)
        symmetric_lattice_vec = symmetric_matrix_to_vector(symmetric_lattice)
        inv_rot_mat = torch.linalg.inv(rot_mat)

        noisy_symmetric_vector, noise = self.lattice_diffusion(
            symmetric_lattice_vec, t_int
        )
        l_t = vector_to_symmetric_matrix(noisy_symmetric_vector)
        noise_matrix = vector_to_symmetric_matrix(noise)
        return l_t, noise_matrix, inv_rot_mat

    def __call__(self, model, batch, t_emb_weights, t_int=None):
        """
        input x has to be cart coords.
        """
        x = batch.pos
        h = batch.x
        lattice = batch.L0
        lattice = lattice.view(-1, 3, 3)
        num_atoms = batch.num_atoms

        x, h = self.normalize(x, h)

        # Sample a timestep t.
        if t_int is None:
            t_int = torch.randint(
                1, self.T + 1, size=(num_atoms.size(0), 1), device=x.device
            ).long()
        else:
            t_int = (
                torch.ones((batch.num_atoms.size(0), 1), device=x.device).long() * t_int
            )
        t_int_atoms = t_int.repeat_interleave(num_atoms, dim=0)

        # Sample noise.
        frac_x_t, target_eps_x, used_sigmas_x = self.pos_diffusion(
            x, t_int_atoms, lattice, num_atoms
        )
        h_t, eps_h = self.type_diffusion(h, t_int_atoms)  # eps is the noise
        l_t, eps_l, inv_rot_mat = self.diffuse_lattice(lattice, t_int)

        # Compute the prediction.
        pred_eps_x, pred_eps_h, pred_eps_l = self.phi(
            frac_x_t,
            h_t,
            t_int_atoms,
            num_atoms,
            l_t,
            model,
            batch,
            t_emb_weights,
            frac=True,
        )

        # Compute the error.
        error_x = self.compute_error(
            pred_eps_x,
            target_eps_x / used_sigmas_x**2,
            batch,
            0.5 * used_sigmas_x**2,
        )  # likelihood reweighting
        error_h = self.compute_error(pred_eps_h, eps_h, batch)

        noisy_symmetric_lattice_hat = torch.matmul(inv_rot_mat, pred_eps_l)
        error_l = self.compute_error_for_global_vec(noisy_symmetric_lattice_hat, eps_l)

        loss = (
            self.cost_coord_coeff * error_x
            + self.cost_type_coeff * error_h
            + self.lattice_coeff * error_l
        )
        return loss.mean()

        # return {
        #     "t": t_int.squeeze(),
        #     "loss": loss.mean(),  # needs to be called "loss" for pytorch lightning to see it.
        #     "coord_loss": error_x.mean(),
        #     "type_loss": error_h.mean(),
        #     "pred_eps_x": pred_eps_x,
        #     "pred_eps_h": pred_eps_h,
        #     "eps_x": target_eps_x,
        #     "eps_h": eps_h,
        # }

    @torch.no_grad()
    def sample(
        self,
        model,
        z_table: AtomicNumberTable,
        t_emb_weights,
        num_atoms: int,
        vis_name: str,
        only_visualize_last: bool,
        show_bonds: bool,
    ):
        num_atomic_states = len(z_table)

        lattice = torch.randn([3, 3]).unsqueeze(0)

        # TODO: verify that we are uing the GPU during inferencing (via nvidia smi)
        # I am not 100% sure that pytorch lightning is using the GPU during inferencing.
        x = (
            torch.randn([num_atoms.sum(), 3], dtype=torch.get_default_dtype())
            * pos_sigma_max
        )
        frac_x = cart_to_frac_coords(x, lattice, num_atoms)
        x = frac_to_cart_coords(frac_x, lattice, num_atoms)

        h = torch.randn([num_atoms.sum(), num_atomic_states])

        for timestep in tqdm(reversed(range(1, self.T))):
            t = torch.full((num_atoms.sum(),), fill_value=timestep)
            timestep_vec = torch.tensor([timestep])  # add a batch dimension

            score_x, score_h, score_l = self.phi(
                frac_x,
                h,
                t,
                num_atoms,
                lattice,
                model,
                Batch(
                    num_atoms=num_atoms, batch=torch.tensor(0).repeat(num_atoms.sum())
                ),
                t_emb_weights,
                frac=True,
            )
            lattice = self.lattice_diffusion.reverse(lattice, score_l, timestep_vec)
            frac_x = self.pos_diffusion.reverse(x, score_x, t, lattice, num_atoms)
            x = frac_to_cart_coords(frac_x, lattice, num_atoms)
            h = self.type_diffusion.reverse(h, score_h, t)

            if (
                not only_visualize_last
                and (timestep != self.T - 1)
                and (timestep % 10 == 0)
            ):
                vis_crystal_during_sampling(
                    z_table, h, lattice, x, vis_name + f"_{timestep}", show_bonds
                )

        x, h = self.unnormalize(
            x, h
        )  # why does mofdiff unnormalize? The fractional x coords can be > 1 after unormalizing.

        output = {
            "x": x,
            "h": h,
            "num_atoms": num_atoms,
            "lattice": lattice,
        }
        vis_crystal_during_sampling(
            z_table, h, lattice, x, vis_name + "_final", show_bonds
        )

        return output
