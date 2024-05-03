import pathlib
from diffusion.diffusion_helpers import (
    VP_limited_mean_and_var,
    polar_decomposition,
    symmetric_matrix_to_vector,
    vector_to_symmetric_matrix,
)
import torch
import os

from diffusion.inference.visualize_lattice import visualize_and_save_lattice

OUT_DIR = f"{pathlib.Path(__file__).parent.resolve()}/../out/vp_limited_mean_and_var"


# we want to sample many lattices at a high time step to see if the lattices look realistic
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    vp = VP_limited_mean_and_var(num_steps=1000, s=0.0001, power=2, clipmax=0.999)
    t = 999  # sample from a very high time step for maximal variance

    square_lattice = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ).unsqueeze(0)

    for i in range(30):
        rotation_matrix, symmetric_matrix = polar_decomposition(square_lattice)
        symmetric_matrix_vector = symmetric_matrix_to_vector(symmetric_matrix)
        num_atoms = torch.tensor([15])
        noisy_symmetric_vector, _symmetric_vector_noise = vp(
            symmetric_matrix_vector, t, num_atoms
        )

        noisy_symmetric_matrix = vector_to_symmetric_matrix(noisy_symmetric_vector)
        noisy_lattice = rotation_matrix @ noisy_symmetric_matrix

        visualize_and_save_lattice(noisy_lattice, f"{OUT_DIR}/{i}.png")
        print(
            f"noisy_symmetric_vector: {noisy_symmetric_vector} noisy_symmetric_matrix: {noisy_symmetric_matrix}"
        )


if __name__ == "__main__":
    main()
