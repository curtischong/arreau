import argparse
import numpy as np
import torch
from diffusion.lattice_dataset import load_dataset
import os

from lightning_wrappers.diffusion import PONITA_DIFFUSION

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
args = parser.parse_args()

torch.set_default_dtype(torch.float64)
model_path = args.model_path
model = PONITA_DIFFUSION.load_from_checkpoint(model_path, strict=False)


num_atoms=torch.tensor([5])
# lattice = np.random.rand(3,3)
dataset = load_dataset("datasets/alexandria_hdf5/10_examples.h5")
sample_L0 = torch.tensor([dataset[0].L0])
# sample_X0 = dataset[0].X0

os.makedirs("out", exist_ok=True)
vis_name = "out/crystal"

model.sample(sample_L0, num_atoms, vis_name, only_visualize_last=True)
