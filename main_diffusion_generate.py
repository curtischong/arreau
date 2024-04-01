import argparse
import torch
from diffusion.inference.create_gif import generate_gif
from diffusion.lattice_dataset import load_dataset
import os

from lightning_wrappers.diffusion import PONITA_DIFFUSION

IMG_DIR = "out"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    return parser.parse_args()

def get_sample_lattice(use_ith_sample_lattice: int):
    # lattice = np.random.rand(3,3)
    dataset = load_dataset("datasets/alexandria_hdf5/10_examples.h5")
    sample_L0 = torch.tensor([dataset[use_ith_sample_lattice].L0])
    # sample_X0 = dataset[0].X0
    return sample_L0

def sample_crystal(Lt, num_atoms):
    # load model
    torch.set_default_dtype(torch.float64)
    model_path = args.model_path
    model = PONITA_DIFFUSION.load_from_checkpoint(model_path, strict=False)

    num_atoms = torch.tensor([num_atoms])

    os.makedirs(IMG_DIR, exist_ok=True)
    vis_name = f"{IMG_DIR}/crystal"

    model.sample(Lt, num_atoms, vis_name, only_visualize_last=False)


if __name__ == "__main__":
    args = parse_args()

    Lt = get_sample_lattice(use_ith_sample_lattice=4)
    sample_crystal(Lt, num_atoms=10)

    generate_gif(src_img_dir=IMG_DIR, output_file=f"{IMG_DIR}/crystal.gif")