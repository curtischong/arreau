import torch

from torch_scatter import scatter
from torch_geometric.data import Batch
import torchmetrics
from tqdm import tqdm
from torch.nn import functional as F

from diffusion.diffusion_helpers import (
    VP,
    VE_pbc,
    cart_to_frac_coords,
    frac_to_cart_coords,
    radius_graph_pbc,
    subtract_cog,
)
from diffusion.lattice_helpers import (
    decode_angles,
    encode_angles,
    lattice_from_params,
    matrix_to_params,
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
        # noisy_symmetric_vec: torch.Tensor,
        model,
        batch: Batch,
        t_emb_weights,
        frac=False,
    ):
        t = self.type_diffusion.betas[t_int].view(-1, 1)
        t_emb = t_emb_weights(t)
        # noisy_symmetric_vec_expanded = torch.repeat_interleave(
        #     noisy_symmetric_vec, num_atoms, dim=0
        # )
        # h_time = torch.cat([h_t, t_emb, noisy_symmetric_vec_expanded], dim=1)

        # encode the lengths and angles for the model
        lattice_lengths_and_angles = matrix_to_params(lattice)
        angles = encode_angles(lattice_lengths_and_angles[:, 3:])
        encoded_lengths_and_angles = torch.cat(
            [lattice_lengths_and_angles[:, :3], angles], dim=-1
        )
        encoded_lengths_and_angles = torch.repeat_interleave(
            encoded_lengths_and_angles, num_atoms, dim=0
        )

        h_time = torch.cat([h_t, t_emb, encoded_lengths_and_angles], dim=1)
        cart_x_t = x_t if not frac else frac_to_cart_coords(x_t, lattice, num_atoms)

        # overwrite the batch with the new values. I'm not making a new batch object since I may miss some attributes.
        # If overwritting leads to problems, we'll need to make a new Batch object
        batch.x = h_time
        batch.pos = cart_x_t
        # batch.vec = torch.repeat_interleave(
        #     noisy_symmetric_matrix, num_atoms, dim=0
        # )  # This line is needed to have each node have it's corresponding lattice vector
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
        pred_eps_h, pred_eps_x, raw_pred_lengths_and_angles, _pred_eps_global_vec = (
            model(batch)
        )

        # normalize the predictions
        used_sigmas_x = self.pos_diffusion.sigmas[t_int].view(-1, 1)
        pred_eps_x = subtract_cog(pred_eps_x, num_atoms)

        pred_lengths = raw_pred_lengths_and_angles[:, :3]
        # pred_lengths = pred_lengths * num_atoms.view(-1, 1).float() ** (1 / 3)
        decoded_angles = decode_angles(raw_pred_lengths_and_angles[:, 3:])
        pred_lengths_and_angles = torch.cat([pred_lengths, decoded_angles], dim=-1)

        return (
            pred_eps_x.squeeze(1) / used_sigmas_x,
            pred_eps_h,
            pred_lengths_and_angles,
        )

    def normalize(self, x, h):
        x = x / self.norm_x
        h = h / self.norm_h
        return x, h

    def unnormalize(self, x, h):
        x = x * self.norm_x
        h = h * self.norm_h
        return x, h

    def lattice_loss2(
        self,
        length_noise: torch.Tensor,
        length_used_sigmas: torch.Tensor,
        angle_noise: torch.Tensor,
        pred_param_noise: torch.Tensor,
    ):
        pred_length_noise = pred_param_noise[:, :3]
        pred_angle_noise = pred_param_noise[:, 3:]

        length_weights = 0.5 * length_used_sigmas**2
        adjusted_length_noise = length_noise / length_used_sigmas**2

        length_loss = (
            length_weights * ((pred_length_noise - adjusted_length_noise) ** 2)
        ).mean()
        angle_loss = F.mse_loss(pred_angle_noise, angle_noise)
        return length_loss + angle_loss

    # screw the frac coords. they're just from 0 to 1. this new lattice will just purturb their cartesian pos a lot
    def diffuse_lattice_params(self, lattice, t_int, num_atoms):
        clean_params = matrix_to_params(lattice)
        clean_lengths = clean_params[:, :3]
        clean_lengths = clean_lengths / num_atoms.view(-1, 1) ** (
            1 / 3
        )  # scale the lengths to the number of atoms

        noisy_lengths, length_noise, length_used_sigmas = self.length_diffusion(
            clean_lengths, t_int
        )

        clean_angles = clean_params[:, 3:]
        noisy_angles, angle_noise = self.angle_diffusion(clean_angles, t_int)

        noisy_params = torch.cat([noisy_lengths, noisy_angles], dim=-1)
        noisy_lattice = lattice_from_params(noisy_params.to(lattice.device))

        return noisy_lattice, length_noise, length_used_sigmas, angle_noise

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
        # noisy_lattice, noise_vec, noisy_symmetric_vec = self.diffuse_lattice(
        #     lattice, t_int
        # )
        noisy_lattice, length_noise, length_used_sigmas, angle_noise = (
            self.diffuse_lattice_params(lattice, t_int, num_atoms)
        )

        # Compute the prediction.
        pred_eps_x, pred_eps_h, pred_param_noise = self.phi(
            frac_x_t,
            h_t,
            t_int_atoms,
            num_atoms,
            noisy_lattice,
            # noisy_symmetric_vec,
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

        # error_l = self.compute_error_for_global_vec(pred_noise_vec, noise_vec)
        # error_l = self.lattice_loss(
        #     pred_lengths_and_angles, noisy_lattice, lattice, num_atoms
        # )
        error_l = self.lattice_loss2(
            length_noise, length_used_sigmas, angle_noise, pred_param_noise
        )

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
            # timestep_vec = torch.tensor([timestep])  # add a batch dimension

            score_x, score_h, score_params = self.phi(
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
            # lattice = self.lattice_diffusion.reverse(lattice, score_l, timestep_vec)
            pred_lengths = score_params[:, :3]
            pred_angles = decode_angles(score_params[:, 3:])
            # pred_lengths = pred_lengths * num_atoms.view(-1, 1).float() ** (1 / 3)
            score_params = torch.cat([pred_lengths, pred_angles], dim=-1)
            next_params = self.lattice_diffusion.reverse(
                score_params, t
            )  # TODO: score fed into model
            lattice = lattice_from_params(next_params)

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
