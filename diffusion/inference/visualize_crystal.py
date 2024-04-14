from enum import Enum
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element
import numpy as np
import plotly.graph_objects as go
from diffusion.inference.predict_bonds import predict_bonds
from diffusion.inference.visualize_lattice import plot_edges, plot_with_parallelopied

from diffusion.tools.atomic_number_table import (
    AtomicNumberTable,
    one_hot_to_atomic_numbers,
)


class VisualizationSetting(Enum):
    NONE = 0
    LAST = 1
    ALL = 2


def element_color(symbol):
    # Dictionary mapping chemical symbols to colors
    color_map = {
        "H": "#F0F8FF",  # Hydrogen
        "He": "#D3D3D3",  # Helium
        "Li": "#B22222",  # Lithium
        "Be": "#9ACD32",  # Beryllium
        "B": "#FFD700",  # Boron
        "C": "#000000",  # Carbon
        "N": "#8F00FF",  # Nitrogen
        "O": "#FF0000",  # Oxygen
        "F": "#DAA520",  # Fluorine
        "Ne": "#EE82EE",  # Neon
        "Na": "#0000FF",  # Sodium
        "Mg": "#228B22",  # Magnesium
        "Al": "#C0C0C0",  # Aluminum
        "Si": "#A52A2A",  # Silicon
        "P": "#FFA500",  # Phosphorus
        "S": "#FFFF00",  # Sulfur
        "Cl": "#00FF00",  # Chlorine
        "Ar": "#FF00FF",  # Argon
        "K": "#F0E68C",  # Potassium
        "Ca": "#E6E6FA",  # Calcium
    }

    # Return the color for the given chemical symbol, or a default color if not found
    return color_map.get(symbol, "#808080")  # Default color is gray


def plot_bonds(fig, structure: Structure):
    bonds = predict_bonds(structure)
    plot_edges(fig, bonds, "#303030")


def vis_crystal_during_sampling(
    z_table: AtomicNumberTable,
    A: np.ndarray,
    lattice: np.ndarray,
    frac_x: np.ndarray,
    name: str,
    show_bonds: bool,
):
    lattice = lattice.squeeze(0)
    atomic_numbers = one_hot_to_atomic_numbers(z_table, A)
    return vis_crystal(atomic_numbers, lattice, frac_x, name, show_bonds)


def vis_crystal(
    atomic_numbers: np.ndarray,
    raw_lattice: np.ndarray,
    frac_x: np.ndarray,
    name: str,
    show_bonds: bool,
):
    fig = plot_crystal(atomic_numbers, raw_lattice, frac_x, show_bonds)
    # Save the plot as a PNG file
    fig.write_image(name + ".png")


def plot_crystal(
    atomic_numbers: np.ndarray,
    raw_lattice: np.ndarray,
    frac_x: np.ndarray,
    show_bonds: bool,
) -> go.Figure:
    lattice = Lattice(raw_lattice)
    element_symbols = [Element.from_Z(z).symbol for z in atomic_numbers]
    pos_arr = []
    for i in range(len(atomic_numbers)):
        pos_arr.append(frac_x[i].tolist())

    # TODO: find a workaround. this is so hacky
    try:
        # https://pymatgen.org/pymatgen.core.html#pymatgen.core.IStructure
        structure = Structure(
            lattice, element_symbols, pos_arr, coords_are_cartesian=False
        )
    except Exception as e:
        print("Error in visualizing crystal", e)
        return

    # Create a Plotly figure
    fig = go.Figure()

    # Add traces for each atom in the structure
    for site in structure.sites:
        atom_type = str(site.specie)
        fig.add_trace(
            go.Scatter3d(
                x=[site.x],
                y=[site.y],
                z=[site.z],
                mode="markers",
                marker=dict(
                    size=5,
                    color=element_color(
                        atom_type
                    ),  # Set the color based on the atom type
                ),
                name=atom_type,
            )
        )
    plot_with_parallelopied(fig, raw_lattice)
    if show_bonds:
        plot_bonds(fig, structure)

    # Set the layout for the 3D plot
    fig.update_layout(
        title="Crystal Structure",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=0),
    )
    return fig
