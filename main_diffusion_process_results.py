from diffusion.diffusion_loss import SampleResult
from diffusion.inference.process_generated_crystals import load_sample_results_from_hdf5
import os

from diffusion.inference.relax import relax
from diffusion.inference.visualize_crystal import vis_crystal

# TODO: put this in a config file?
OUT_DIR = "out"
RELAX_DIR = f"{OUT_DIR}/relax"


def get_one_crystal(sample_result: SampleResult, sample_idx: int):
    lattice = sample_result.lattice[sample_idx]
    crystal_start_idx = sample_result.idx_start[sample_idx]
    num_atoms = sample_result.num_atoms[sample_idx]
    end_idx = crystal_start_idx + num_atoms
    frac_x = sample_result.frac_x[crystal_start_idx:end_idx]
    atomic_numbers = sample_result.atomic_numbers[crystal_start_idx:end_idx]
    return lattice, frac_x, atomic_numbers


def relax_one_crystal(sample_result: SampleResult, sample_idx: int):
    os.makedirs(RELAX_DIR, exist_ok=True)
    lattice, frac_x, atomic_numbers = get_one_crystal(sample_result, sample_idx)
    relax(lattice, frac_x, atomic_numbers, RELAX_DIR)


def visualize_one_crystal(sample_result: SampleResult, sample_idx: int):
    lattice, frac_x, atomic_numbers = get_one_crystal(sample_result, sample_idx)
    name = f"{OUT_DIR}/crystal_{sample_idx}"
    vis_crystal(atomic_numbers, lattice, frac_x, name, show_bonds=False)


if __name__ == "__main__":
    sample_results = load_sample_results_from_hdf5("out/crystals.h5")
    visualize_one_crystal(sample_results, sample_idx=1)
