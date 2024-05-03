import subprocess
import time

from diffusion.lattice_dataset import CrystalDataset
import numpy as np
import torch

from diffusion.prep_datasets import save_dataset


def rotate_lattice_about_origin(lattice):
    lower_south_west_corner = torch.min(lattice, axis=0).values
    lattice -= lower_south_west_corner
    rotation_matrix = np.array(
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    )  # 90 degrees about the x axis
    rotated_lattice = np.dot(lattice, rotation_matrix)
    return rotated_lattice


# I should do inference on a low timstep since hte noise isn't as high
def prep_rotated_datasets():
    # "datasets/alexandria_hdf5/alexandria_ps_000_take1_rotated.h5"
    # dataset = "datasets/alexandria_hdf5/alexandria_ps_000_take10.h5"
    dataset = CrystalDataset(
        [
            "datasets/alexandria_hdf5/alexandria_ps_000_take1.h5",
        ]
    )
    ith_sample = 0
    lattice = dataset[ith_sample].L0
    frac_x = dataset[ith_sample].X0
    atomic_numbers = dataset[ith_sample].A0
    # fig = plot_crystal(atomic_numbers, lattice, frac_x, show_bonds=False)
    # fig.show()

    rotated_lattice = rotate_lattice_about_origin(lattice)
    # fig = plot_crystal(atomic_numbers, rotated_lattice, frac_x, show_bonds=False)
    # fig.show()
    save_dataset(
        "alexandria_ps_000_take1_rotated",
        [atomic_numbers],
        np.expand_dims(rotated_lattice, axis=0),
        [frac_x],
    )


def main():
    # prep_rotated_datasets()
    start_time = time.time()
    subprocess.run(
        [
            "python",
            "main_diffusion.py",
            "--num_timesteps=1000",
            "--epochs=10000",
            "--gpus=0",
            "--radius=5",
            "--num_workers=-1",
            "--max_neighbors=8",
            "--batch_size=10",
            "--dataset=eval-equivariance",
        ]
    )
    end_time = time.time()
    print(f"Time: {end_time - start_time}")


if __name__ == "__main__":
    main()
