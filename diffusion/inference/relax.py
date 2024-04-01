import pathlib
from ase import units
from ase.md.langevin import Langevin
from ase.io import read, write

from mace.calculators import MACECalculator
from diffusion.inference.visualize_crystal import vis_crystal
from diffusion.lattice_dataset import load_dataset
from ase import Atoms
from pymatgen.core.periodic_table import Element
from ase.optimize import BFGS

def get_sample_system():
    # lattice = np.random.rand(3,3)
    dataset = load_dataset("datasets/alexandria_hdf5/10_examples.h5")
    system_sample = dataset[1]
    return system_sample.L0, system_sample.X0, system_sample.atomic_numbers

def relax(out_dir):
    num_relaxations = 20

    # get the inital system
    L0, X, atomic_numbers = get_sample_system()
    symbols = [Element.from_Z(z).symbol for z in atomic_numbers]

    positions = X @ L0
    for i in range(num_relaxations):
        system = Atoms(symbols=symbols, positions=positions, pbc=(True,True,True))

        # create the calculator
        model_path = f"{pathlib.Path(__file__).parent.resolve()}/../../models/2024-01-07-mace-128-L2_epoch-199.model"
        system.calc = MACECalculator(model_paths=model_path, device='cpu')

        # Perform the relaxation for one timestep
        dyn = BFGS(system)
        dyn.run(fmax=0.05, steps=1)

        positions = system.get_positions()
        vis_crystal(atomic_numbers, L0, X, f"{out_dir}/relax_{i}")

if __name__ == "__main__":
    relax()