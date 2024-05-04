# This file converts the alexandria datasets to hdf5 format so it's faster to load
# These hdf5 files are also more efficient since we drop unused columns
from multiprocessing import Process
from pymatgen.entries.computed_entries import ComputedStructureEntry
import numpy as np
import h5py
import json
import bz2
import pathlib
import os


ROOT_PATH = f"{pathlib.Path(__file__).parent.resolve()}/.."
IN_DIR = f"{ROOT_PATH}/datasets/alexandria"
OUT_DIR = f"{ROOT_PATH}/datasets/alexandria_hdf5"


def normalize_lattice(lattice):
    # The reason why we want to normalize the lattice is because if you convert the lengths / angles of a lattice into the matrix form, there are more than one valid matrix forms (since it's a basis)
    # To make our functions consistent when converting a lattice from params to matrix form, we first convert all lattices in the dataset to params first.
    # However, if you think about it, learning the lattice matrix isn't good. we should probably learn the angles / lengths.
    # if you think aboout it some more, a lattice is just a human-defined basis. we could technically make due by representing everything as cubes, although, there may be multiple lattice cells in one cube.
    # so I guess learning a lattice is better (reduces number of atoms needed in the system). One example is if the lattice parallelepied are REALLY slanted, so one cartesian cube has 10 parallelepied cells in it (a waste of atoms to simulate).

    # original_lattice = torch.tensor(structure.lattice.matrix).unsqueeze(0)
    # new_lattice = lattice_from_params(matrix_to_params(original_lattice)).squeeze(0)
    # lattice_matrices[idx] = new_lattice.detach().cpu().numpy()

    # inv_lattice = torch.linalg.pinv(new_lattice)
    # cart_coords = torch.tensor(structure.cart_coords)
    # frac_coords_in_new_lattice = cart_coords @ inv_lattice
    # # I used the below code to verify that the caresian coords are the same in the new lattice
    # # cart_coords_in_new_lattice = frac_coords_in_new_lattice @ new_lattice
    # frac_coords_arrays.append(frac_coords_in_new_lattice.detach().cpu().numpy())
    pass


def prep_data_and_save_hdf5(filename, take_max_num_examples=None):
    print(f"prepping {filename}")
    with bz2.open(f"{IN_DIR}/{filename}.json.bz2", "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    if take_max_num_examples is not None:
        filename = f"{filename}_take{take_max_num_examples}"
        data["entries"] = data["entries"][:take_max_num_examples]

    entries = [ComputedStructureEntry.from_dict(i) for i in data["entries"]]
    print(f"Found {len(entries)} entries for {filename}")

    atomic_number_vectors = []
    lattice_matrices = np.zeros((len(entries), 3, 3))
    frac_coords_arrays = []
    for idx, entry in enumerate(entries):
        structure = entry.structure
        atomic_number_vector = np.empty(len(structure.species), dtype=int)
        for i, species in enumerate(structure.species):
            atomic_number_vector[i] = species.Z
        atomic_number_vectors.append(atomic_number_vector)

        # new_lattice = normalize_lattice(lattice)

        lattice_matrices[idx] = structure.lattice.matrix
        frac_coords_arrays.append(structure.frac_coords)

    save_dataset(filename, atomic_number_vectors, lattice_matrices, frac_coords_arrays)


def save_dataset(filename, atomic_number_vectors, lattice_matrices, frac_coords_arrays):
    # Save the data to an HDF5 file
    os.makedirs(OUT_DIR, exist_ok=True)
    with h5py.File(f"{OUT_DIR}/{filename}.h5", "w") as f:
        atom_onehot_group = f.create_group("atomic_number")
        for i, vector in enumerate(atomic_number_vectors):
            atom_onehot_group.create_dataset(str(i), data=vector)

        f.create_dataset("lattice_matrix", data=lattice_matrices)

        frac_coords_group = f.create_group("frac_coord")
        for i, array in enumerate(frac_coords_arrays):
            frac_coords_group.create_dataset(str(i), data=array)


def prep_10_examples():
    prep_data_and_save_hdf5("alexandria_ps_000", take_max_num_examples=10)


def main():
    prep_10_examples()  # this prepares a small sample dataset so when we train locally, it's fast.

    NUM_FILES = 5

    processes = []
    for i in range(NUM_FILES):
        file_name = f"alexandria_ps_00{i}"

        # https://stackoverflow.com/questions/55529319/how-to-create-multiple-threads-dynamically-in-python
        p = Process(target=prep_data_and_save_hdf5, args=(file_name,))
        p.start()
        processes.append(p)

    # Wait all processes to finish.
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
