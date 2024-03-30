import torch

from torch_scatter import scatter
from torch_geometric.data import Batch

from diffusion.diffusion_helpers import VP, GaussianFourierProjection, VE_pbc, frac_to_cart_coords, subtract_cog


pos_sigma_min = 0.001
pos_sigma_max = 10.

type_power = 2
type_clipmax = 0.999
fourier_scale = 16
t_emb_dim = 64

class DiffusionLoss:
    def __init__(self, num_timesteps):
        self.T = num_timesteps
        self.pos_diffusion = VE_pbc(
            self.T,
            sigma_min=pos_sigma_min,
            sigma_max=pos_sigma_max
        )

        self.type_diffusion = VP(
            self.T,
            power=type_power,
            clipmax=type_clipmax,
        )

        self.cost_coord_coeff = 1
        self.cost_type_coeff = 1
        self.norm_x = 10.
        self.norm_h = 10.
        self.t_emb = GaussianFourierProjection(
            t_emb_dim // 2, fourier_scale
        )

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


    def phi(self, x_t, h_t, t_int, num_atoms, lattice, model, batch: Batch, frac=False):
        t = self.type_diffusion.betas[t_int].view(-1, 1)
        # t_emb = self.t_emb(t)
        # h_time = torch.cat([h_t, t_emb], dim=1)
        # frac_x_t = x_t if frac else cart_to_frac_coords(x_t, lattice, num_atoms)
        cart_x_t = x_t if not frac else frac_to_cart_coords(x_t, lattice, num_atoms)
        batch.x = h_t
        batch.pos = cart_x_t
        pred_eps_x, pred_eps_h = model(
            x = h_t,
            frac_coord_variance=t,
            # num_atoms=num_atoms,
            positions=cart_x_t,
        )
        used_sigmas_x = self.pos_diffusion.sigmas[t_int].view(-1, 1)
        pred_eps_x = subtract_cog(pred_eps_x, num_atoms)
        return pred_eps_x / used_sigmas_x, pred_eps_h
    
    def normalize(self, x, h, lengths):
        x = x / self.norm_x
        lengths = lengths / self.norm_x
        h = h / self.norm_h
        return x, h, lengths
    
    def __call__(self, model, batch, t_int=None):
        """
        input x has to be cart coords.
        """
        x = batch.pos
        h = batch.x
        lattice = batch.L0
        lattice = lattice.view(-1, 3, 3)
        num_atoms = batch.num_atoms


        x, h, lattice = self.normalize(x, h, lattice)

        # Sample a timestep t.
        if t_int is None:
            t_int = torch.randint(
                1, self.T + 1, size=(num_atoms.size(0), 1), device=x.device
            ).long()
        else:
            t_int = (
                torch.ones((batch.num_atoms.size(0), 1), device=x.device).long() * t_int
            )
        t_int = t_int.repeat_interleave(num_atoms, dim=0)

        # Sample noise.
        frac_x_t, target_eps_x, used_sigmas_x = self.pos_diffusion(
            x, t_int, lattice, num_atoms
        )
        h_t, eps_h = self.type_diffusion(h, t_int)

        # Compute the prediction.
        pred_eps_x, pred_eps_h = self.phi(
            frac_x_t, h_t, t_int, num_atoms, lattice, model, batch, frac=True
        )

        # Compute the error.
        error_x = self.compute_error(
            pred_eps_x,
            target_eps_x / used_sigmas_x**2,
            batch,
            0.5 * used_sigmas_x**2,
        )  # likelihood reweighting
        error_h = self.compute_error(pred_eps_h, eps_h, batch)

        loss = self.hparams.cost_coord * error_x + self.hparams.cost_type * error_h

        return {
            "t": t_int.squeeze(),
            "diffusion_loss": loss.mean(),
            "coord_loss": error_x.mean(),
            "type_loss": error_h.mean(),
            "pred_eps_x": pred_eps_x,
            "pred_eps_h": pred_eps_h,
            "eps_x": target_eps_x,
            "eps_h": eps_h,
        }