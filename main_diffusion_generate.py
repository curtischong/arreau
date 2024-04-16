import argparse
from typing import Optional
import torch
from diffusion.diffusion_loss import SampleResult
from diffusion.inference.create_gif import generate_gif
from diffusion.inference.process_generated_crystals import save_sample_results_to_hdf5
from diffusion.inference.visualize_crystal import VisualizationSetting
from lightning_wrappers.diffusion import PONITA_DIFFUSION
import numpy as np

OUT_DIR = "out"
DIFFUSION_DIR = f"{OUT_DIR}/diffusion"
SHOW_BONDS = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model file"
    )
    return parser.parse_args()


def get_model() -> PONITA_DIFFUSION:
    args = parse_args()
    # load model
    torch.set_default_dtype(torch.float64)
    model_path = args.model_path
    return PONITA_DIFFUSION.load_from_checkpoint(model_path, strict=False)


def generate_single_crystal(
    num_atoms: int,
    visualization_setting: VisualizationSetting,
    use_constant_atomic_symbols: Optional[list[str]],
):
    model = get_model()
    result = model.sample(
        num_atoms_per_sample=num_atoms,
        num_samples_in_batch=1,
        visualization_setting=visualization_setting,
        show_bonds=SHOW_BONDS,
        use_constant_atomic_symbols=use_constant_atomic_symbols,
    )
    if visualization_setting != VisualizationSetting.NONE:
        generate_gif(src_img_dir=DIFFUSION_DIR, output_file=f"{OUT_DIR}/crystal.gif")

    result.idx_start = np.array([0])
    save_sample_results_to_hdf5(result, f"{OUT_DIR}/crystals.h5")


def generate_n_crystals(
    num_crystals: int,
    num_atoms_per_sample: int,
    use_constant_atomic_symbols: Optional[list[str]],
):
    num_crystals_per_batch = 8
    assert num_crystals_per_batch > 0
    assert num_crystals_per_batch <= num_crystals
    assert (
        num_crystals % num_crystals_per_batch == 0
    ), f"num_crystals ({num_crystals}) must be divisible by num_crystals_per_batch ({num_crystals_per_batch})"
    total_num_atoms = num_crystals * num_atoms_per_sample

    model = get_model()

    crystals = SampleResult()
    crystals.frac_x = np.empty((total_num_atoms, 3))
    crystals.atomic_numbers = np.empty((total_num_atoms))
    crystals.lattice = np.empty((num_crystals, 3, 3))
    crystals.idx_start = np.arange(0, total_num_atoms, num_atoms_per_sample)
    crystals.num_atoms = np.full(num_crystals, num_atoms_per_sample)

    for i in range(0, num_crystals, num_crystals_per_batch):
        generated_crystals = model.sample(
            num_atoms_per_sample=num_atoms_per_sample,
            num_samples_in_batch=num_crystals_per_batch,
            visualization_setting=VisualizationSetting.NONE,
            show_bonds=SHOW_BONDS,
            use_constant_atomic_symbols=use_constant_atomic_symbols,
        )
        batch_start = i * num_atoms_per_sample
        batch_end = (i + num_crystals_per_batch) * num_atoms_per_sample
        crystals.frac_x[batch_start:batch_end] = generated_crystals.frac_x
        crystals.atomic_numbers[batch_start:batch_end] = (
            generated_crystals.atomic_numbers
        )
        crystals_batch_start = i
        crystals_batch_end = i + num_crystals_per_batch
        crystals.lattice[crystals_batch_start:crystals_batch_end] = (
            generated_crystals.lattice
        )

    save_sample_results_to_hdf5(crystals, f"{OUT_DIR}/crystals.h5")


if __name__ == "__main__":
    num_atoms = 8
    # use_constant_atomic_symbols = ["Ac", "Ac", "Ir", "Ag"]
    use_constant_atomic_symbols = None

    if use_constant_atomic_symbols is not None:
        num_atoms = len(use_constant_atomic_symbols)

    # generate_single_crystal(
    #     num_atoms=num_atoms,
    #     visualization_setting=VisualizationSetting.NONE,
    #     use_constant_atomic_symbols=use_constant_atomic_symbols,
    # )
    generate_n_crystals(
        num_crystals=32,
        num_atoms_per_sample=6,
        use_constant_atomic_symbols=use_constant_atomic_symbols,
    )
