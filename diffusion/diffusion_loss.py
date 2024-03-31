import torch

from torch_scatter import scatter
from torch_geometric.data import Batch
import torchmetrics
import numpy as np
from tqdm import tqdm

from diffusion.diffusion_helpers import VP, VE_pbc, cart_to_frac_coords, frac_to_cart_coords, subtract_cog
from diffusion.tools.atomic_number_table import AtomicNumberTable
from diffusion.visualize_crystal import vis_crystal


pos_sigma_min = 0.001
pos_sigma_max = 10.

type_power = 2
type_clipmax = 0.999

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
    def __init__(self, num_timesteps):
        super().__init__()
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
        # self.norm_x = 10. # I'm not sure why mofdiff uses these values. I'm going to use 1.0 for now.
        # self.norm_h = 10.
        self.norm_x = 1.
        self.norm_h = 1.

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


    def phi(self, x_t, h_t, t_int, num_atoms, lattice, model, batch: Batch, t_emb_weights, frac=False):
        t = self.type_diffusion.betas[t_int].view(-1, 1)
        t_emb = t_emb_weights(t)
        h_time = torch.cat([h_t, t_emb], dim=1)
        cart_x_t = x_t if not frac else frac_to_cart_coords(x_t, lattice, num_atoms)

        # overwrite the batch with the new values. I'm not making a new batch object since I may miss some attributes.
        # If overwritting leads to problems, we'll need to make a new Batch object
        batch.x = h_time
        batch.pos = cart_x_t

        # compute the predictions
        pred_eps_h, pred_eps_x = model(batch)

        # normalize the predictions
        used_sigmas_x = self.pos_diffusion.sigmas[t_int].view(-1, 1)
        pred_eps_x = subtract_cog(pred_eps_x, num_atoms)
        return pred_eps_x.squeeze(1) / used_sigmas_x, pred_eps_h
    
    def normalize(self, x, h):
        x = x / self.norm_x
        h = h / self.norm_h
        return x, h
    
    def unnormalize(self, x, h):
        x = x * self.norm_x
        h = h * self.norm_h
        return x, h
    
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
        t_int = t_int.repeat_interleave(num_atoms, dim=0)

        # Sample noise.
        frac_x_t, target_eps_x, used_sigmas_x = self.pos_diffusion(
            x, t_int, lattice, num_atoms
        )
        h_t, eps_h = self.type_diffusion(h, t_int)

        # Compute the prediction.
        pred_eps_x, pred_eps_h = self.phi(
            frac_x_t, h_t, t_int, num_atoms, lattice, model, batch, t_emb_weights, frac=True
        )

        # Compute the error.
        error_x = self.compute_error(
            pred_eps_x,
            target_eps_x / used_sigmas_x**2,
            batch,
            0.5 * used_sigmas_x**2,
        )  # likelihood reweighting
        error_h = self.compute_error(pred_eps_h, eps_h, batch)

        loss = self.cost_coord_coeff * error_x + self.cost_type_coeff * error_h

        return {
            "t": t_int.squeeze(),
            "loss": loss.mean(), # needs to be called "loss" for pytorch lightning to see it.
            "coord_loss": error_x.mean(),
            "type_loss": error_h.mean(),
            "pred_eps_x": pred_eps_x,
            "pred_eps_h": pred_eps_h,
            "eps_x": target_eps_x,
            "eps_h": eps_h,
        }

    @torch.no_grad()
    def sample(
        self,
        model,
        z_table: AtomicNumberTable,
        lattice: np.ndarray,
        t_emb_weights,
        num_atoms: int,
        vis_name: str,
        only_visualize_last: bool,
        save_freq=False
    ):
        num_atomic_states = len(z_table)
        # TODO: verify that we are uing the GPU during inferencing (via nvidia smi)
        # I am not 100% sure that pytorch lightning is using the GPU during inferencing.
        x = (
            torch.randn([num_atoms.sum(), 3], dtype=torch.get_default_dtype())
            * pos_sigma_max
        )
        frac_x = cart_to_frac_coords(x, lattice, num_atoms)
        x = frac_to_cart_coords(frac_x, lattice, num_atoms)

        h = torch.randn([num_atoms.sum(), num_atomic_states])

        if save_freq:
            all_x = [x.clone().cpu()]
            all_h = [h.clone().cpu()]

        for timestep in tqdm(reversed(range(1, self.T))):
            t = torch.full((num_atoms.sum(),), fill_value=timestep)

            score_x, score_h = self.phi(
                frac_x, h, t, num_atoms, lattice, model, Batch(), t_emb_weights, frac=True
            )
            frac_x = self.pos_diffusion.reverse(
                x, score_x, t, lattice, num_atoms
            )
            x = frac_to_cart_coords(frac_x, lattice, num_atoms)
            h = self.type_diffusion.reverse(h, score_h, t)

            if save_freq and (t[0] % save_freq == 0):
                all_x.append(x.clone().cpu())
                all_h.append(h.clone().cpu())
            
            if not only_visualize_last and (timestep != self.T-1) and (timestep % 10 == 0):
                vis_crystal(z_table, h, lattice, x, vis_name + f"_{timestep}")

        if save_freq:
            all_x.append(x.clone().cpu())
            all_h.append(h.clone().cpu())

        x, h = self.unnormalize(x, h) # why does mofdiff unnormalize? The fractional x coords can be > 1 after unormalizing.

        output = {
            "x": x,
            "h": h,
            "num_atoms": num_atoms,
            "lattice": lattice,
        }
        vis_crystal(z_table, h, lattice, x, vis_name + "_final")

        if save_freq:
            output.update(
                {
                    "all_x": torch.stack(all_x, dim=0),
                    "all_h": torch.stack(all_h, dim=0),
                }
            )
        return output