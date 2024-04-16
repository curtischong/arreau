import pathlib
from diffusion.inference.visualize_crystal import (
    visualize_and_save_crystal,
)
import torch
from diffusion.lattice_dataset import CrystalDataset
import os

OUT_DIR = f"{pathlib.Path(__file__).parent.resolve()}/../out"
dataset_vis_dir = f"{OUT_DIR}/alexandria_vis"


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
    os.makedirs(dataset_vis_dir, exist_ok=True)

    for i in range(50):
        print(f"sample {i}")
        ith_sample = dataset[i]

        atomic_num = torch.argmax(ith_sample.A0, dim=1)
        lattice = ith_sample.L0.numpy()
        frac_x = ith_sample.X0.numpy()
        visualize_and_save_crystal(
            atomic_num,
            lattice,
            frac_x,
            name=f"{dataset_vis_dir}/{i}",
            show_bonds=False,
        )


if __name__ == "__main__":
    main()
