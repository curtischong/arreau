# density = mass / volume
# density = num atoms / volume of lattice
# to find the constnat c for the mean of the variance preservation of the lattice diffusion, we need to find the average density of the dataset


import pathlib
from diffusion.lattice_dataset import load_data
import h5py
import random

DATA_DIR = f"{pathlib.Path(__file__).parent.resolve()}/../datasets/alexandria_hdf5"


def main():
    datasets = [
        "alexandria_ps_000",
        "alexandria_ps_001",
        "alexandria_ps_002",
        "alexandria_ps_003",
        "alexandria_ps_004",
        # "alexandria_ps_000_take10"
    ]

    # I'm giving up on unique lattices. they have very similar ones
    # unique_lattices = set()
    lattices = []

    for dataset in datasets:
        atomic_number_vector, lattice_matrix, _frac_x = load_data(
            f"{DATA_DIR}/{dataset}.h5"
        )
        for i in range(len(lattice_matrix)):
            lattices.append(lattice_matrix[i])

    random.shuffle(lattices)

    with h5py.File(DATA_DIR + "/known_lattices.h5", "w") as file:
        group = file.create_group("lattices")
        group.create_dataset("lattices", data=lattices)


if __name__ == "__main__":
    main()
