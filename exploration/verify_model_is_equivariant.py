import subprocess
import time

from diffusion.lattice_dataset import CrystalDataset
from diffusion.inference.visualize_crystal import plot_crystal
import numpy as np
import torch


def rotate_lattice_about_origin(lattice, atomic_coords):
    lower_south_west_corner = torch.min(lattice, axis=0).values
    lattice -= lower_south_west_corner
    # atomic_coords -= lower_south_west_corner
    rotation_matrix = np.array(
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    )  # 90 degrees about the x axis
    rotated_lattice = np.dot(lattice, rotation_matrix)
    # rotated_x = np.dot(atomic_coords, rotation_matrix)
    return (rotated_lattice, atomic_coords)


# I should do inference on a low timstep since hte noise isn't as high
def prep_rotated_datasets():
    # "datasets/alexandria_hdf5/alexandria_ps_000_take1_rotated.h5"
    # dataset = "datasets/alexandria_hdf5/alexandria_ps_000_take10.h5"
    dataset = CrystalDataset(
        [
            "datasets/alexandria_hdf5/alexandria_ps_000_take10.h5",
        ]
    )
    ith_sample = 0
    lattice = dataset[ith_sample].L0
    frac_x = dataset[ith_sample].X0
    atomic_numbers = dataset[ith_sample].A0
    fig = plot_crystal(atomic_numbers, lattice, frac_x, show_bonds=False)
    fig.show()

    rotated_lattice, rotated_x = rotate_lattice_about_origin(lattice, frac_x)
    fig = plot_crystal(atomic_numbers, rotated_lattice, rotated_x, show_bonds=False)
    fig.show()


def main():
    prep_rotated_datasets()
    return
    start_time = time.time()
    subprocess.run(
        [
            "python",
            "main_diffusion.py",
            "--num_timesteps=10",
            "--epochs=100",
            "--gpus=0",
            "--radius=5",
            "--num_workers=-1",
            "--max_neighbors=8",
            "--batch_size=10",
            "--is_local_dev=true",
        ]
    )
    end_time = time.time()
    print(f"Time: {end_time - start_time}")


if __name__ == "__main__":
    main()
