import pathlib
from ase import units
from ase.md.langevin import Langevin
from ase.io import read, write

from mace.calculators import MACECalculator

from lattice_dataset import load_dataset

def get_sample_system():
    # lattice = np.random.rand(3,3)
    dataset = load_dataset("datasets/alexandria_hdf5/10_examples.h5")
    return dataset.L0, dataset.X0, dataset.atomic_numbers

def relax():
    sample_system = get_sample_system()


    model_path = f"{pathlib.Path(__file__).parent.resolve()}/../../models/2024-01-07-mace-128-L2_epoch-199.model"
    calculator = MACECalculator(model_path=model_path, device='cpu')
    init_conf = read('BOTNet-datasets/dataset_3BPA/test_300K.xyz', '0')
    init_conf.set_calculator(calculator)

    dyn = Langevin(init_conf, 0.5*units.fs, temperature_K=310, friction=5e-3)
    def write_frame():
            dyn.atoms.write('md_3bpa.xyz', append=True)
    dyn.attach(write_frame, interval=50)
    dyn.run(100)

if __name__ == "__main__":
    relax()