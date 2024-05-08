# density = mass / volume
# density = num atoms / volume of lattice
# to find the constnat c for the mean of the variance preservation of the lattice diffusion, we need to find the average density of the dataset


import pathlib
from diffusion.lattice_dataset import load_data
from diffusion.lattice_helpers import matrix_to_params
import torch

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

    unique_angles = set()

    for dataset in datasets:
        atomic_number_vector, lattice_matrix, _frac_x = load_data(
            f"{DATA_DIR}/{dataset}.h5"
        )
        for i in range(len(atomic_number_vector)):
            lengths, angles = matrix_to_params(
                torch.tensor(lattice_matrix[i]).unsqueeze(0)
            )
            for angle in angles[0]:
                unique_angles.add(angle.item())

    print(list(unique_angles))
    # Average density: 0.05539856385043283
    # Average volume: 152.51649752530176


if __name__ == "__main__":
    main()
