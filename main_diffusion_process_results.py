# The purpose of this file is to take the generated crystals (saved in the .h5 file) and do further processing on them (visualization, relaxation, etc)

from diffusion.diffusion_loss import SampleResult
from diffusion.inference.process_generated_crystals import (
    get_one_crystal,
    load_sample_results_from_hdf5,
)
import os

from diffusion.inference.relax import relax
from diffusion.inference.visualize_crystal import visualize_and_save_crystal

# TODO: put this in a config file?
OUT_DIR = "out"
RELAX_DIR = f"{OUT_DIR}/relax"


def relax_one_crystal(sample_result: SampleResult, sample_idx: int):
    os.makedirs(RELAX_DIR, exist_ok=True)
    lattice, frac_x, atomic_numbers = get_one_crystal(sample_result, sample_idx)
    relax(lattice, frac_x, atomic_numbers, RELAX_DIR)


def visualize_one_crystal(sample_result: SampleResult, sample_idx: int):
    lattice, frac_x, atomic_numbers = get_one_crystal(sample_result, sample_idx)
    name = f"{OUT_DIR}/crystal_{sample_idx}"
    visualize_and_save_crystal(atomic_numbers, lattice, frac_x, name, show_bonds=False)


if __name__ == "__main__":
    sample_results = load_sample_results_from_hdf5("out/crystals.h5")
    # visualize_one_crystal(sample_results, sample_idx=1)
    relax_one_crystal(sample_results, sample_idx=1)
