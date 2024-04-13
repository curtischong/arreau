import plotly.graph_objects as go
import numpy as np
import torch


def plot_edges(fig, edges, color):
    for edge in edges:
        fig.add_trace(
            go.Scatter3d(
                x=[edge[0][0], edge[1][0]],
                y=[edge[0][1], edge[1][1]],
                z=[edge[0][2], edge[1][2]],
                mode="lines",
                line=dict(color=color, width=5),
                showlegend=False,  # Do not add to the legend
            )
        )


def plot_with_parallelopied(fig, L):
    v1 = L[0]
    v2 = L[1]
    v3 = L[2]
    # Create the parallelepiped by combining the basis vectors
    points = np.array([[0, 0, 0], v1, v1 + v2, v2, v3, v1 + v3, v1 + v2 + v3, v2 + v3])

    # Create the edges of the parallelepiped as tuples of Cartesian coordinates
    edges = [
        (tuple(points[0]), tuple(points[1])),
        (tuple(points[1]), tuple(points[2])),
        (tuple(points[2]), tuple(points[3])),
        (tuple(points[3]), tuple(points[0])),
        (tuple(points[4]), tuple(points[5])),
        (tuple(points[5]), tuple(points[6])),
        (tuple(points[6]), tuple(points[7])),
        (tuple(points[7]), tuple(points[4])),
        (tuple(points[0]), tuple(points[4])),
        (tuple(points[1]), tuple(points[5])),
        (tuple(points[2]), tuple(points[6])),
        (tuple(points[3]), tuple(points[7])),
    ]
    # Plot the edges using the helper function
    plot_edges(fig, edges, "#0d5d85")

    return points


def visualize_lattice(lattice: torch.Tensor, out_path: str):
    # Create a Plotly figure
    fig = go.Figure()
    points = plot_with_parallelopied(fig, lattice.squeeze(0))
    smallest = np.min(points, axis=0)
    largest = np.max(points, axis=0)

    # Set the layout for the 3D plot
    fig.update_layout(
        title="Crystal Structure",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[smallest[0], largest[0]]),
            yaxis=dict(range=[smallest[1], largest[1]]),
            zaxis=dict(range=[smallest[2], largest[2]]),
        )
    )

    # If the lattices look really tiny, you're probably looking at them from the worng angle
    # This moves the camera to the eye level so you can check to see how the lattices really look
    # camera = dict(eye=dict(x=2, y=2, z=0.1))
    # fig.update_layout(scene_camera=camera)

    # Save the plot as a PNG file
    fig.write_image(out_path)
    print(f"Saved {out_path}")
