import pathlib

from diffusion.inference.visualize_crystal import visualize_and_save_crystal
from diffusion.lattice_dataset import load_dataset
from ase import Atoms
from ase.optimize import BFGS
import numpy as np
from mace.calculators import MACECalculator


def get_sample_system():
    dataset = load_dataset("datasets/alexandria_hdf5/10_examples.h5")
    system_sample = dataset[1]
    return system_sample.L0, system_sample.X0, system_sample.atomic_numbers


def relax(L0: np.ndarray, frac_x: np.ndarray, atomic_numbers: np.ndarray, out_dir: str):
    model_path = f"{pathlib.Path(__file__).parent.resolve()}/../../models/2024-01-07-mace-128-L2_epoch-199.model"
    calculator = MACECalculator(model_paths=model_path, device="cpu")  # noqa: F821

    # set initial positions
    # for i in range(num_relaxations):
    # positions += np.random.randn(*positions.shape) * 0.5
    system = Atoms(
        numbers=atomic_numbers,
        scaled_positions=frac_x,
        cell=L0,
        pbc=(True, True, True),
    )

    # create the calculator
    system.calc = calculator

    # Perform the relaxation for one timestep
    dyn = BFGS(system)
    dyn.run()

    frac_x = dyn.atoms.get_scaled_positions()
    visualize_and_save_crystal(
        atomic_numbers, L0, frac_x, f"{out_dir}/relax_final", show_bonds=False
    )
    return frac_x


if __name__ == "__main__":
    current_directory = pathlib.Path(__file__).parent.resolve()
    L0, frac_x, atomic_numbers = get_sample_system()
    relax(L0, frac_x, atomic_numbers, f"{current_directory}/../../out")
