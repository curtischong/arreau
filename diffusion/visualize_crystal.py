from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element
import numpy as np
import plotly.graph_objects as go

def plot_with_parallelopied(fig, L):
    v1 = L[0]
    v2 = L[1]
    v3 = L[2]

    # Create the parallelepiped by combining the basis vectors
    points = np.array([[0, 0, 0],
                       v1,
                       v1 + v2,
                       v2,
                       v3,
                       v1 + v3,
                       v1 + v2 + v3,
                       v2 + v3])

    # Create the edges of the parallelepiped
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]

    # Add the edges to the plot
    for edge in edges:
        fig.add_trace(
            go.Scatter3d(
                x=points[[edge[0], edge[1]], 0],
                y=points[[edge[0], edge[1]], 1],
                z=points[[edge[0], edge[1]], 2],
                mode='lines',
                line=dict(color='blue', width=5),
                showlegend=False  # Do not add to the legend
            )
        )

def element_color(symbol):
    # Dictionary mapping chemical symbols to colors
    color_map = {
        "H": "#F0F8FF",   # Hydrogen
        "He": "#D3D3D3",  # Helium
        "Li": "#B22222",  # Lithium
        "Be": "#9ACD32",  # Beryllium
        "B": "#FFD700",   # Boron
        "C": "#000000",   # Carbon
        "N": "#8F00FF",   # Nitrogen
        "O": "#FF0000",   # Oxygen
        "F": "#DAA520",   # Fluorine
        "Ne": "#EE82EE",  # Neon
        "Na": "#0000FF",  # Sodium
        "Mg": "#228B22",  # Magnesium
        "Al": "#C0C0C0",  # Aluminum
        "Si": "#A52A2A",  # Silicon
        "P": "#FFA500",   # Phosphorus
        "S": "#FFFF00",   # Sulfur
        "Cl": "#00FF00",  # Chlorine
        "Ar": "#FF00FF",  # Argon
        "K": "#F0E68C",   # Potassium
        "Ca": "#E6E6FA"   # Calcium
    }

    # Return the color for the given chemical symbol, or a default color if not found
    return color_map.get(symbol, "#808080")  # Default color is gray

def vis_crystal(A, L_t, X, name):
    lattice = Lattice(L_t)
    element_symbols = [Element.from_Z(z).symbol for z in A]
    pos_arr = []
    for i in range(len(A)):
        pos_arr.append(X[i].tolist())

    # https://pymatgen.org/pymatgen.core.html#pymatgen.core.IStructure
    structure = Structure(lattice, element_symbols, pos_arr, coords_are_cartesian=False)
  
    # Create a Plotly figure
    fig = go.Figure()

    # Add traces for each atom in the structure
    for site in structure.sites:
        atom_type = str(site.specie)
        fig.add_trace(go.Scatter3d(
            x=[site.x],
            y=[site.y],
            z=[site.z],
            mode='markers',
            marker=dict(
                size=5,
                color=element_color(atom_type)  # Set the color based on the atom type
            ),
            name=atom_type
        ))
    plot_with_parallelopied(fig, L_t)

    # Set the layout for the 3D plot
    fig.update_layout(
        title='Crystal Structure',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    # Save the plot as a PNG file
    fig.write_image(name)