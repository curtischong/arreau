import pathlib
from diffusion.diffusion_loss import SampleResult
from diffusion.inference.process_generated_crystals import (
    get_crystal_indexes,
    get_one_crystal,
)

from diffusion.lattice_dataset import load_dataset
from ase import Atoms
from ase.optimize import BFGS
import numpy as np
from mace.calculators import MACECalculator


def get_sample_system():
    dataset = load_dataset("datasets/alexandria_hdf5/10_examples.h5")
    system_sample = dataset[1]
    return system_sample.L0, system_sample.X0, system_sample.atomic_numbers


def relax(lattice: np.ndarray, frac_x: np.ndarray, atomic_numbers: np.ndarray):
    model_path = f"{pathlib.Path(__file__).parent.resolve()}/../../models/2024-01-07-mace-128-L2_epoch-199.model"
    calculator = MACECalculator(model_paths=model_path, device="cpu")  # noqa: F821

    # set initial positions
    system = Atoms(
        numbers=atomic_numbers,
        scaled_positions=frac_x,
        cell=lattice,
        pbc=(True, True, True),
    )

    # create the calculator
    system.calc = calculator

    # Perform the relaxation for one timestep
    dyn = BFGS(system)
    dyn.run()

    frac_x = dyn.atoms.get_scaled_positions()
    # visualize_and_save_crystal(
    #     atomic_numbers, L0, frac_x, f"{out_dir}/relax_final", show_bonds=False
    # )
    return frac_x


# TODO: find a way to parallelize this
# Since mace trains in batches, we should be able to batch this as well
def bulk_relax(sample_result: SampleResult) -> SampleResult:
    num_samples = len(sample_result.num_atoms)
    new_sample_result = SampleResult(
        frac_x=np.empty(sample_result.frac_x.shape),
        atomic_numbers=sample_result.atomic_numbers.copy(),
        lattice=sample_result.lattice.copy(),
        num_atoms=sample_result.num_atoms.copy(),
        idx_start=sample_result.idx_start.copy(),
    )

    for i in range(num_samples):
        lattice, frac_x, atomic_numbers = get_one_crystal(sample_result, i)
        new_frac_x = relax(
            lattice=lattice,
            frac_x=frac_x,
            atomic_numbers=atomic_numbers,
        )

        crystal_start_idx, end_idx = get_crystal_indexes(sample_result, i)

        new_sample_result.frac_x[crystal_start_idx:end_idx] = new_frac_x
    return new_sample_result


if __name__ == "__main__":
    L0, frac_x, atomic_numbers = get_sample_system()
    relax(L0, frac_x, atomic_numbers)
