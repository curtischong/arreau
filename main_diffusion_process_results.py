from diffusion.diffusion_loss import SampleResult
from diffusion.inference.process_generated_crystals import load_sample_results_from_hdf5
import os

from diffusion.inference.relax import relax

# TODO: put this in a config file?
OUT_DIR = "out"
RELAX_DIR = f"{OUT_DIR}/relax"


def relax_one_crystal(sample_result: SampleResult, sample_idx: int):
    os.makedirs(RELAX_DIR, exist_ok=True)
    lattice = sample_result.lattice[sample_idx]

    crystal_start_idx = sample_result.idx_start[sample_idx]
    num_atoms = sample_result.num_atoms[sample_idx]
    end_idx = crystal_start_idx + num_atoms
    x = sample_result.x[crystal_start_idx:end_idx]
    h = sample_result.atomic_numbers[crystal_start_idx:end_idx]
    relax(lattice, x, h, RELAX_DIR)


def visualize_one_crystal(sample_result: SampleResult):
    # vis_crystal(z_table, L_t, X, name, show_bonds)
    pass


if __name__ == "__main__":
    sample_results = load_sample_results_from_hdf5("out/crystals.h5")
    relax_one_crystal(sample_results, sample_idx=2)
