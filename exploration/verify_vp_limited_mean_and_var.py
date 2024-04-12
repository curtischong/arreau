import pathlib
from diffusion.diffusion_helpers import (
    VP_limited_mean_and_var,
    polar_decomposition,
    symmetric_matrix_to_vector,
    vector_to_symmetric_matrix,
)
import torch
import os
import plotly.graph_objects as go

from diffusion.inference.visualize_crystal import plot_with_parallelopied

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
        noisy_symmetric_vector, _symmetric_vector_noise = vp(symmetric_matrix_vector, t)

        noisy_symmetric_matrix = vector_to_symmetric_matrix(noisy_symmetric_vector)
        noisy_lattice = rotation_matrix @ noisy_symmetric_matrix

        # Create a Plotly figure
        fig = go.Figure()
        plot_with_parallelopied(fig, noisy_lattice.squeeze(0))

        # Set the layout for the 3D plot
        fig.update_layout(
            title="Crystal Structure",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            margin=dict(l=0, r=0, b=0, t=0),
        )

        # Save the plot as a PNG file
        fig.write_image(f"{OUT_DIR}/{i}.png")
        print(f"Saved {i} in {OUT_DIR}")


if __name__ == "__main__":
    main()
