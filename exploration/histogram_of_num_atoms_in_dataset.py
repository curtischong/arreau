from diffusion.lattice_dataset import CrystalDataset
from collections import defaultdict


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

    atoms_cnt = defaultdict(int)
    for i in range(len(dataset.configs)):
        num_atoms = len(dataset.configs[i].atomic_numbers)
        atoms_cnt[num_atoms] += 1

    def num_datapoints_above_n_atoms(n):
        cnt = 0
        for key, value in atoms_cnt.items():
            if key > n:
                cnt += value
        return cnt

    print(dict(atoms_cnt))
    return


if __name__ == "__main__":
    main()
