from dataclasses import dataclass
from typing import Optional
import torch

from torch_geometric.data import Batch
import torchmetrics
from tqdm import tqdm
import numpy as np
from diffusion.d3pm import D3PM

from diffusion.diffusion_helpers import (
    VE_pbc,
    VP_lattice,
    frac_to_cart_coords,
    radius_graph_pbc,
    sample_bravais_angles,
)
from diffusion.lattice_helpers import lattice_from_params, matrix_to_params
from diffusion.tools.atomic_number_table import (
    AtomicNumberTable,
    atomic_number_indexes_to_atomic_numbers,
)
from diffusion.inference.visualize_crystal import (
    VisualizationSetting,
    vis_crystal_during_sampling,
)
from torch.nn import functional as F


pos_sigma_min = 0.001
pos_sigma_max = 1.0  # this was originally 10 but since we're diffusing over frac coords now, I changed it to 1

type_power = 2
lattice_power = 2
type_clipmax = 0.999
lattice_clipmax = 0.999

avg_num_atoms_plus_ghost_atoms = 20


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
    def __init__(self, args, z_table: AtomicNumberTable):
        super().__init__()
        num_atomic_states = len(z_table)
        self.cutoff = args.radius
        self.max_neighbors = args.max_neighbors
        self.T = args.num_timesteps
        self.pos_diffusion = VE_pbc(
            self.T, sigma_min=pos_sigma_min, sigma_max=pos_sigma_max
        )

        self.d3pm = D3PM(
            x0_model=None,
            n_T=args.num_timesteps,
            num_classes=num_atomic_states,
            forward_type="mask",
        )

        self.lattice_diffusion = VP_lattice(
            num_steps=self.T,
            power=lattice_power,
            clipmax=lattice_clipmax,
        )
        self.num_atomic_states = num_atomic_states

        self.coord_loss_weight = 1
        self.atom_type_loss_weight = 3
        self.lattice_loss_weight = 1
        self.ghost_atom_index = z_table.z_to_index(
            AtomicNumberTable.GHOST_ATOMIC_NUMBER
        )

    def compute_frac_x_error(
        self, pred_frac_eps_x, target_frac_eps_x, ghost_atom_indices
    ):
        # Clamping between 0-1 is really important to avoid problems from numerical instabilities
        distance_abs_diff = torch.clamp(
            torch.remainder((pred_frac_eps_x - target_frac_eps_x).abs(), 1),
            min=0,
            max=1,
        )
        distance_abs_diff[ghost_atom_indices] = (
            0  # ghost atoms should not contribute to the loss since their positions are arbitrary
        )
        # PERF: maybe we can make this cheaper by doing this before the clamp (only get the remainder and clamp the indices that matter)
        # but array lengths will be dynamic and execution time may be varied, which is harder for speculative execution maybe?
        # There'll also be more allocations, so it may not be faster

        # This is the key thing: when working in mod 1, the distance between 0.1 and 0.9 is NOT 0.8. It's 0.2
        distance_wrapped_diff = torch.min(distance_abs_diff, 1 - distance_abs_diff)

        # the squared euclidean distance between each point
        distance_wrapped_diff_squared = distance_wrapped_diff**2
        squared_euclidean_dist = torch.sum(distance_wrapped_diff_squared, dim=1)

        return torch.mean(squared_euclidean_dist)

    def predict_scores(
        self,
        noisy_frac_x: torch.Tensor,
        noisy_atom_types: torch.Tensor,
        t_int: torch.Tensor,
        num_atoms: torch.Tensor,
        lattice: torch.Tensor,
        model,
        batch: Batch,
        t_emb_weights,
    ):
        t = self.lattice_diffusion.betas[t_int].view(-1, 1)
        t_emb = t_emb_weights(t)

        num_atoms_feat = torch.repeat_interleave(num_atoms, num_atoms, dim=0).unsqueeze(
            -1
        )
        lengths, angles = matrix_to_params(lattice)
        lengths_feat = torch.repeat_interleave(lengths, num_atoms, dim=0)
        angles_feat = torch.repeat_interleave(angles, num_atoms, dim=0)
        scaled_lengths = (
            lengths / num_atoms.unsqueeze(-1)
        ).abs()  # take the abs to prevent imaginary numbers
        scaled_lengths_feat = torch.repeat_interleave(scaled_lengths, num_atoms, dim=0)

        scalar_feats = torch.cat(
            [
                noisy_atom_types,
                t_emb,
                num_atoms_feat,
                lengths_feat,
                angles_feat,
                scaled_lengths_feat,
            ],
            dim=1,
        )
        noisy_cart_x = frac_to_cart_coords(noisy_frac_x, lattice, num_atoms)

        lattice_feat = torch.repeat_interleave(lattice, num_atoms, dim=0)

        # overwrite the batch with the new values. I'm not making a new batch object since I may miss some attributes.
        # If overwritting leads to problems, we'll need to make a new Batch object
        batch.x = scalar_feats
        batch.pos = noisy_cart_x
        batch.vec = torch.cat([noisy_frac_x.unsqueeze(1), lattice_feat], dim=1)

        # we need to overwrite the edge_index for the batch since when we add noise to the positions, some atoms may be
        # so far apart from each other they are no longer considered neighbors. So we need to recompute the neighbors.

        # I'm not sure how useful neighbors is. It's a count of how many neighbors each atom has
        edge_index, cell_offsets, neighbors, inter_atom_distance, neighbor_direction = (
            radius_graph_pbc(
                noisy_cart_x,
                lattice,
                batch.num_atoms,
                self.cutoff,
                self.max_neighbors,
                device=noisy_cart_x.device,
                remove_self_edges=True,  # Removing self-loops since the embedding after message passing is also dependent on the embedding of the current node. see https://github.com/curtischong/arreau/pull/97 for details
            )
        )
        batch.edge_index = edge_index
        batch.dists = inter_atom_distance  # TODO: rename dists to inter_atom_distance
        batch.inter_atom_direction = neighbor_direction
        batch.lattice = lattice

        batch.batch_of_edge = batch.batch[batch.edge_index[0]]

        # compute the predictions
        (
            predicted_atom_type_0_logits,
            pred_frac_eps_x,
            _pred_lengths_0,
            _global_output_vector,
            _pred_edge_distance_score,
        ) = model(batch)

        return (
            pred_frac_eps_x.squeeze(
                1
            ),  # squeeze 1 since the only per-node vector output is the frac coords, so there is a useless dimension.
            predicted_atom_type_0_logits,
        )

    def diffuse_lattice_params(self, lattice: torch.Tensor, t_int: torch.Tensor):
        lengths, angles = matrix_to_params(lattice)
        noisy_lengths, _lengths_noise = self.lattice_diffusion(lengths, t_int)
        return (noisy_lengths, lengths, angles)

    def num_ghost_atoms_to_add(self, batch: Batch) -> torch.Tensor:
        num_atoms = batch.num_atoms
        diff = avg_num_atoms_plus_ghost_atoms - num_atoms

        # Generate random values from a normal distribution with mean=diff and std=8
        normal_values = torch.round(
            torch.randn(diff.shape[0], device=diff.device) * 8 + diff
        ).int()

        # we never want to add less than 5 ghost atoms
        num_ghost_atoms = torch.max(torch.tensor(5), normal_values)

        new_num_atoms = num_atoms + num_ghost_atoms
        return num_atoms, num_ghost_atoms, new_num_atoms

    def get_atom_and_ghost_atom_indices(
        self,
        num_atoms: torch.Tensor,
        new_num_atoms: torch.Tensor,
        num_ghost_atoms: torch.Tensor,
    ):
        zero_tensor = torch.tensor([0], device=num_atoms.device)
        num_atom_starts = torch.cat([zero_tensor, num_atoms[:-1].cumsum(dim=0)])
        new_num_atom_starts = torch.cat([zero_tensor, new_num_atoms[:-1].cumsum(dim=0)])

        # Calculate the indexes of the atoms / ghost atoms in the new array
        # Note: we cannot just "append the ghost atoms to the end of the array" since we assume that the num_atoms (for each batch) is consecutive
        num_atoms_before = torch.repeat_interleave(num_atom_starts, num_atoms)
        new_num_atoms_before = torch.repeat_interleave(new_num_atom_starts, num_atoms)
        all_indices = torch.arange(num_atoms.sum(), device=num_atoms.device)

        # the main idea is here. we subtract each index by the cumulative number of atoms in the previous batch. this will give us the index of the atom in the current batch
        original_index_in_batch = all_indices - num_atoms_before
        # then we just offset each index by its new position in the new array
        atom_indices = original_index_in_batch + new_num_atoms_before

        num_ghost_atoms_before = torch.cat(
            [zero_tensor, num_ghost_atoms[:-1].cumsum(0)]
        )
        all_ghost_indices = torch.arange(
            num_ghost_atoms.sum(), device=num_atoms.device
        ) - torch.repeat_interleave(num_ghost_atoms_before, num_ghost_atoms)
        # the main idea is here. the ghost atoms just start "num_atoms" after the last REAL atom in the current batch
        ghost_atom_starts = new_num_atom_starts + num_atoms
        ghost_atom_offset = torch.repeat_interleave(ghost_atom_starts, num_ghost_atoms)
        ghost_atom_indices = all_ghost_indices + ghost_atom_offset
        return atom_indices, ghost_atom_indices

    # TODO: move these helper functions into a separate file in a new directory
    def add_ghost_atoms(self, batch: Batch):
        # should the positions of the ghost atoms be accounted for in the frac_x loss?
        # I don't think so since their positions are arbitrary.
        # this means that it's really important to predict the original atomic type. since that determines the ghost atom's affect on the loss??
        num_atoms, num_ghost_atoms, new_num_atoms = self.num_ghost_atoms_to_add(batch)
        atom_indices, ghost_atom_indices = self.get_atom_and_ghost_atom_indices(
            num_atoms, new_num_atoms, num_ghost_atoms
        )

        total_num_atoms = new_num_atoms.sum()
        total_num_ghost_atoms = num_ghost_atoms.sum()

        # now populate the new arrays
        new_frac_x_0 = torch.zeros(
            (total_num_atoms, 3),
            dtype=torch.get_default_dtype(),
            device=num_atoms.device,
        )
        new_frac_x_0[atom_indices] = batch.X0
        new_frac_x_0[ghost_atom_indices] = (
            (
                torch.randn(total_num_ghost_atoms, 3, device=num_atoms.device) * 10
            )  # * 10 just to make sure it's more random
            % 1
        )

        # TODO: make this dtype an int? I think we eed to make batch.A0 an int
        new_atom_type_0 = torch.zeros(
            (total_num_atoms,), dtype=torch.long, device=num_atoms.device
        )
        new_atom_type_0[atom_indices] = batch.A0
        new_atom_type_0[ghost_atom_indices] = torch.full(
            (total_num_ghost_atoms,),
            self.ghost_atom_index,
            dtype=torch.long,
            device=num_atoms.device,
        )

        batch_identifiers = torch.repeat_interleave(
            torch.arange(num_atoms.shape[0], device=num_atoms.device), new_num_atoms
        )

        return (
            new_num_atoms,
            new_frac_x_0,
            new_atom_type_0,
            ghost_atom_indices,
            batch_identifiers,
        )

    def __call__(self, model, batch, t_emb_weights, t_int=None):
        lattice_0 = batch.L0
        lattice_0 = lattice_0.view(-1, 3, 3)
        num_atoms, frac_x_0, atom_type_0, ghost_atom_indices, batch.batch = (
            self.add_ghost_atoms(batch)
        )

        # Sample a timestep t.
        # TODO: can we simplify this? is t_int always None? Verification code may inconsistently pass in t_int vs train code
        if t_int is None:
            t_int = torch.randint(
                1, self.T + 1, size=(num_atoms.size(0), 1), device=frac_x_0.device
            ).long()
        else:
            t_int = (
                torch.ones((batch.num_atoms.size(0), 1), device=frac_x_0.device).long()
                * t_int
            )
        noisy_int_atoms = t_int.repeat_interleave(num_atoms, dim=0)

        # Sample noise.
        noisy_frac_x, target_frac_eps_x, used_sigmas_x = self.pos_diffusion(
            frac_x_0, noisy_int_atoms, lattice_0, num_atoms
        )
        noisy_atom_type = self.d3pm.get_xt(atom_type_0, noisy_int_atoms.squeeze())

        noisy_atom_type_onehot = F.one_hot(noisy_atom_type, self.num_atomic_states)

        # Compute the prediction.
        (pred_frac_eps_x, predicted_atom_type_0_logits) = self.predict_scores(
            noisy_frac_x,
            noisy_atom_type_onehot,
            noisy_int_atoms,
            num_atoms,
            lattice_0,
            model,
            batch,
            t_emb_weights,
        )
        # TODO: we softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        # or we just argmax. but either way, we get the atomic type of the ghost atoms

        # Compute the error.
        error_frac_x = self.compute_frac_x_error(
            pred_frac_eps_x,
            target_frac_eps_x,
            ghost_atom_indices,
            # 0.5 * used_sigmas_x**2,
        )  # likelihood reweighting

        error_atomic_type = self.d3pm.calculate_loss(
            atom_type_0,
            predicted_atom_type_0_logits,
            noisy_atom_type,
            noisy_int_atoms.squeeze(),
        )
        # do not divide lengths by num_atoms. since we do NOT know how many REAL atoms are in the system during inferencing

        loss = (
            self.coord_loss_weight * error_frac_x
            + (
                (
                    self.atom_type_loss_weight + self.coord_loss_weight
                )  # we need to add the coord loss weight since the type loss can eliminate the coord loss by predicting all ghost types
                * error_atomic_type
            )
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
        # if we want to test if the model can make carbon crystals, we tell it to only diffuse on carbon atoms
        constant_atoms: Optional[torch.Tensor] = None,
    ) -> SampleResult:
        num_atomic_states = len(z_table)

        # tip: use this page to see what variance to use in your normal distributions https://homepage.divms.uiowa.edu/~mbognar/applets/normal.html
        angles = torch.tensor(
            [sample_bravais_angles("monoclinic") for _ in range(num_samples_in_batch)]
        )
        lengths = torch.randn([num_samples_in_batch, 3])

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

        if constant_atoms is not None:
            atom_types = constant_atoms
        else:
            # init as the mask state
            atom_types = torch.full(
                (num_samples_in_batch * num_atoms_per_sample,), num_atomic_states - 1
            )

        for timestep in tqdm(reversed(range(1, self.T))):
            t = torch.full((num_atoms.sum(),), fill_value=timestep)
            timestep_vec = torch.tensor([timestep])  # add a batch dimension

            score_frac_x, score_atom_types, pred_lengths_0 = self.predict_scores(
                frac_x,
                F.one_hot(atom_types, num_atomic_states),
                t,
                num_atoms,
                lengths,
                angles,
                model,
                Batch(
                    num_atoms=num_atoms,
                    batch=torch.arange(0, num_samples_in_batch).repeat_interleave(
                        num_atoms_per_sample
                    ),
                ),
                t_emb_weights,
            )
            lengths = self.lattice_diffusion.reverse_given_x0(
                lengths, pred_lengths_0, timestep_vec
            )
            lattice = lattice_from_params(lengths, angles)

            frac_x = self.pos_diffusion.reverse(
                frac_x, score_frac_x, t, lattice, num_atoms
            )
            atom_types = self.d3pm.reverse(atom_types, score_atom_types, t)
            if constant_atoms is not None:
                atom_types = constant_atoms

            if (timestep != self.T - 1) and (
                (
                    visualization_setting == VisualizationSetting.ALL
                    and (timestep % 10 == 0)
                )
                or (visualization_setting == VisualizationSetting.ALL_DETAILED)
            ):
                vis_crystal_during_sampling(
                    z_table,
                    atom_types,
                    lattice,
                    frac_x,
                    vis_name + f"_{timestep}",
                    show_bonds,
                )

        if visualization_setting != VisualizationSetting.NONE:
            vis_crystal_during_sampling(
                z_table, atom_types, lattice, frac_x, vis_name + "_final", show_bonds
            )
        atomic_numbers = atomic_number_indexes_to_atomic_numbers(z_table, atom_types)
        return SampleResult(
            num_atoms=num_atoms.numpy(),
            frac_x=frac_x.numpy(),
            atomic_numbers=atomic_numbers,
            lattice=lattice.numpy(),
        )
