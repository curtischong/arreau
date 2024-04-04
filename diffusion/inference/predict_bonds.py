from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN


# returns a list of tuples of cartesian coordinates of the bonds
def predict_bonds(structure: Structure):
    nn = CrystalNN()
    bonded_structure = nn.get_bonded_structure(structure)

    coords = []
    for edge in bonded_structure.graph.edges():
        nodes = bonded_structure.graph.nodes
        site1, site2 = edge
        coords1 = nodes[site1]["coords"]
        coords2 = nodes[site2]["coords"]
        coords.append((coords1, coords2))
    return coords
