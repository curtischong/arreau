import argparse
import numpy as np
from diffusion.lattice_dataset import load_dataset

from lightning_wrappers.diffusion import PONITA_DIFFUSION

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
args = parser.parse_args()

model_path = args.model_path
model = PONITA_DIFFUSION.load_from_checkpoint(model_path, strict=False)

num_atoms=5
lattice = np.random.rand(3,3)
dataset = load_dataset("/Users/curtischong/Documents/dev/lucera/data/processed_data.h5")
sample_L0 = dataset[0].L0
sample_X0 = dataset[0].X0
print(model.sample())
