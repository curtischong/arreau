# The purpose of this file is to take the generated crystals (saved in the .h5 file) and do further processing on them (visualization, relaxation, etc)

from diffusion.diffusion_loss import SampleResult
from diffusion.inference.process_generated_crystals import (
    get_one_crystal,
    load_sample_results_from_hdf5,
    save_sample_results_to_hdf5,
)
import os

from diffusion.inference.relax import bulk_relax, relax
from diffusion.inference.visualize_crystal import visualize_and_save_crystal
import numpy as np

# TODO: put this in a config file?
OUT_DIR = "out"
RELAX_DIR = f"{OUT_DIR}/relax"


def relax_one_crystal(sample_result: SampleResult, sample_idx: int):
    os.makedirs(RELAX_DIR, exist_ok=True)
    lattice, frac_x, atomic_numbers = get_one_crystal(sample_result, sample_idx)
    new_frac_x = relax(lattice, frac_x, atomic_numbers)

    relaxed_results = SampleResult(
        frac_x=new_frac_x,
        atomic_numbers=atomic_numbers,
        lattice=np.expand_dims(lattice, axis=0),
        idx_start=np.array([0]),
        num_atoms=np.array([sample_result.num_atoms[sample_idx]]),
    )
    save_sample_results_to_hdf5(relaxed_results, f"{RELAX_DIR}/relaxed.h5")


def visualize_one_crystal(sample_result: SampleResult, sample_idx: int):
    lattice, frac_x, atomic_numbers = get_one_crystal(sample_result, sample_idx)
    name = f"{OUT_DIR}/crystal_{sample_idx}"
    visualize_and_save_crystal(atomic_numbers, lattice, frac_x, name, show_bonds=False)


def relax_all_crystals(sample_result: SampleResult):
    new_sample_result = bulk_relax(sample_result)
    save_sample_results_to_hdf5(new_sample_result, f"{RELAX_DIR}/relaxed.h5")


if __name__ == "__main__":
    sample_results = load_sample_results_from_hdf5("out/crystals.h5")
    # visualize_one_crystal(sample_results, sample_idx=1)
    # relax_one_crystal(sample_results, sample_idx=1)
    relax_all_crystals(sample_results)
