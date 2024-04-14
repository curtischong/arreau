from diffusion.inference.process_generated_crystals import (
    get_one_crystal,
    load_sample_results_from_hdf5,
)
from diffusion.inference.visualize_crystal import plot_crystal


OUT_DIR = "out"


def main():
    crystal_file = f"{OUT_DIR}/crystals.h5"
    sample_result = load_sample_results_from_hdf5(crystal_file)
    lattice, frac_x, atomic_numbers = get_one_crystal(sample_result, 0)

    # shift the crystal to make the atoms near the center of the unit cell
    # frac_x = (frac_x + np.array([0.3, 0.6, 0.2])) % 1

    fig = plot_crystal(atomic_numbers, lattice, frac_x, show_bonds=False)
    fig.show()


if __name__ == "__main__":
    main()
