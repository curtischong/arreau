# density = mass / volume
# density = num atoms / volume of lattice
# to find the constnat c for the mean of the variance preservation of the lattice diffusion, we need to find the average density of the dataset


import pathlib
import numpy as np
from diffusion.lattice_dataset import load_data


DATA_DIR = f"{pathlib.Path(__file__).parent.resolve()}/../datasets/alexandria_hdf5"


def main():
    datasets = [
        "alexandria_ps_000",
        "alexandria_ps_001",
        "alexandria_ps_002",
        "alexandria_ps_003",
        "alexandria_ps_004",
    ]

    total_density = 0.0
    total_volume = 0.0
    num_samples = 0

    for dataset in datasets:
        atomic_number_vector, lattice_matrix, _frac_x = load_data(
            f"{DATA_DIR}/{dataset}.h5"
        )
        for i in range(len(atomic_number_vector)):
            volume = np.linalg.det(lattice_matrix[i])
            atomic_numbers = atomic_number_vector[i]
            total_density += len(atomic_numbers) / volume
            total_volume += volume
            num_samples += 1

    print(f"Average density: {total_density / num_samples}")
    print(f"Average volume: {total_volume / num_samples}")
    # Average density: 0.05539856385043283


if __name__ == "__main__":
    main()
