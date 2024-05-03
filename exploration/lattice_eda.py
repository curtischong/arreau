from diffusion.inference.visualize_lattice import visualize_and_save_lattice
from diffusion.lattice_dataset import CrystalDataset
import os

LATTICE_EDA_OUT_DIR = "out/lattice_eda"


def main():
    dataset = CrystalDataset(
        [
            "datasets/alexandria_hdf5/alexandria_ps_000.h5",
            # "datasets/alexandria_hdf5/alexandria_ps_001.h5",
            # "datasets/alexandria_hdf5/alexandria_ps_002.h5",
            # "datasets/alexandria_hdf5/alexandria_ps_003.h5",
            # "datasets/alexandria_hdf5/alexandria_ps_004.h5",
        ]
    )
    os.makedirs(LATTICE_EDA_OUT_DIR, exist_ok=True)
    for i in range(50):
        data = dataset[i]
        visualize_and_save_lattice(data.L0, f"{LATTICE_EDA_OUT_DIR}/{i}.png")


if __name__ == "__main__":
    main()
