import pathlib

from diffusion.inference.visualize_crystal import visualize_and_save_crystal
from diffusion.lattice_dataset import load_dataset
from ase import Atoms
from ase.optimize import BFGS
import numpy as np
from ase.calculators.lj import LennardJones


def get_sample_system():
    dataset = load_dataset("datasets/alexandria_hdf5/10_examples.h5")
    system_sample = dataset[1]
    return system_sample.L0, system_sample.X0, system_sample.atomic_numbers


def relax(L0: np.ndarray, frac_x: np.ndarray, atomic_numbers: np.ndarray, out_dir: str):
    num_relaxations = 5

    model_path = f"{pathlib.Path(__file__).parent.resolve()}/../../models/2024-01-07-mace-128-L2_epoch-199.model"
    # calculator = MACECalculator(model_paths=model_path, device="cpu")
    calculator = LennardJones(
        epsilon=0.01042, sigma=3.4, maxiter=1000
    )  # we need a high iter so it converges

    # symbols = [Element.from_Z(z).symbol for z in atomic_numbers]

    # set initial positions
    positions = frac_x @ L0
    for i in range(num_relaxations):
        # positions += np.random.randn(*positions.shape) * 0.5
        system = Atoms(
            numbers=atomic_numbers, positions=positions, pbc=(True, True, True)
        )

        # create the calculator
        system.calc = calculator

        # Perform the relaxation for one timestep
        dyn = BFGS(system)
        dyn.run()

        positions = system.get_positions()
        visualize_and_save_crystal(
            atomic_numbers, L0, frac_x, f"{out_dir}/relax_{i}", show_bonds=False
        )


if __name__ == "__main__":
    current_directory = pathlib.Path(__file__).parent.resolve()
    L0, frac_x, atomic_numbers = get_sample_system()
    relax(L0, frac_x, atomic_numbers, f"{current_directory}/../../out")
