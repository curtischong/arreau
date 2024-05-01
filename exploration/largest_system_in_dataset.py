from diffusion.lattice_dataset import CrystalDataset


def main():
    dataset = CrystalDataset(
        [
            # "datasets/alexandria_hdf5/alexandria_ps_000_take10.h5",
            "datasets/alexandria_hdf5/alexandria_ps_000.h5",
            "datasets/alexandria_hdf5/alexandria_ps_001.h5",
            "datasets/alexandria_hdf5/alexandria_ps_002.h5",
            "datasets/alexandria_hdf5/alexandria_ps_003.h5",
            "datasets/alexandria_hdf5/alexandria_ps_004.h5",
        ]
    )

    most_atoms = 0
    most_atoms_config = None
    most_atoms_config_idx = 0
    for i in range(len(dataset.configs)):
        num_atoms = len(dataset.configs[i].atomic_numbers)
        if num_atoms > most_atoms:
            most_atoms = num_atoms
            most_atoms_config = dataset.configs[i]
            most_atoms_config_idx = i

    print(f"Most atoms: {most_atoms}")
    print(
        f"Most atoms config: {most_atoms_config.X0}, {most_atoms_config.L0}, {most_atoms_config.atomic_numbers}"
    )
    print(f"Most atoms config idx: {most_atoms_config_idx}")


if __name__ == "__main__":
    main()
