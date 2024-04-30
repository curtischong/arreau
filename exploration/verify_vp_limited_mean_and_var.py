import pathlib
from diffusion.diffusion_helpers import (
    VP_lattice,
    polar_decomposition,
    symmetric_matrix_to_vector,
    vector_to_symmetric_matrix,
)
import torch
import os
from diffusion.inference.visualize_lattice import visualize_multiple_lattices

from diffusion.lattice_helpers import matrix_to_params

OUT_DIR = f"{pathlib.Path(__file__).parent.resolve()}/../out/vp_limited_mean_and_var"


# we want to sample many lattices at a high time step to see if the lattices look realistic
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    vp = VP_lattice(num_steps=1000, s=0.0001, power=2, clipmax=0.999)  # noqa: F821
    t = 800  # sample from a very high time step for maximal variance

    real_lattice = torch.tensor(
        [
            [7.864230632781982, -0.028291359543800354, 0.010975549928843975],
            [-2.929015636444092, 7.298478603363037, -0.010975549928843975],
            [-2.7713515758514404, -3.428903341293335, 6.512240409851074],
        ]
    ).unsqueeze(0)

    num_samples = 10000
    all_angles = []
    for i in range(num_samples):
        rotation_matrix, symmetric_matrix = polar_decomposition(real_lattice)
        symmetric_matrix_vector = symmetric_matrix_to_vector(symmetric_matrix)
        num_atoms = torch.tensor([8])
        noisy_symmetric_vector, _symmetric_vector_noise = vp(
            symmetric_matrix_vector, t, num_atoms
        )

        noisy_symmetric_matrix = vector_to_symmetric_matrix(noisy_symmetric_vector)
        noisy_lattice = rotation_matrix @ noisy_symmetric_matrix
        params = matrix_to_params(noisy_lattice)
        angles = params[:, 3:]
        all_angles.append(angles.squeeze())

        visualize_multiple_lattices([real_lattice, noisy_lattice], f"{OUT_DIR}/{i}.png")
    quantiles = torch.quantile(
        torch.stack(all_angles), torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    )
    print("quantiles", quantiles)


if __name__ == "__main__":
    main()
