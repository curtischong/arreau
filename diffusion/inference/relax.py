import pathlib

from mace.calculators import MACECalculator
from diffusion.inference.visualize_crystal import vis_crystal
from diffusion.lattice_dataset import load_dataset
from ase import Atoms
from pymatgen.core.periodic_table import Element
from ase.optimize import BFGS
import numpy as np


def get_sample_system():
    dataset = load_dataset("datasets/alexandria_hdf5/10_examples.h5")
    system_sample = dataset[1]
    return system_sample.L0, system_sample.X0, system_sample.atomic_numbers


def relax(L0: np.ndarray, frac_x: np.ndarray, atomic_numbers: np.ndarray, out_dir: str):
    num_relaxations = 5

    model_path = f"{pathlib.Path(__file__).parent.resolve()}/../../models/2024-01-07-mace-128-L2_epoch-199.model"
    calculator = MACECalculator(model_paths=model_path, device="cpu")

    symbols = [Element.from_Z(z).symbol for z in atomic_numbers]

    # set initial positions
    positions = frac_x @ L0
    for i in range(num_relaxations):
        system = Atoms(symbols=symbols, positions=positions, pbc=(True, True, True))

        # create the calculator
        system.calc = calculator

        # Perform the relaxation for one timestep
        dyn = BFGS(system)
        dyn.run(fmax=0.05, steps=5)

        positions = system.get_positions()
        vis_crystal(
            atomic_numbers, L0, frac_x, f"{out_dir}/relax_{i}", show_bonds=False
        )


if __name__ == "__main__":
    current_directory = pathlib.Path(__file__).parent.resolve()
    L0, frac_x, atomic_numbers = get_sample_system()
    relax(L0, frac_x, atomic_numbers, f"{current_directory}/../../out")
