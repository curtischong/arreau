import pathlib
import h5py
from diffusion.diffusion_loss import SampleResult

from diffusion.tools.atomic_number_table import AtomicNumberTable


def save_sample_results_to_hdf5(crystals: SampleResult, filename: str):
    with h5py.File(filename, "w") as file:
        group = file.create_group("crystals")
        group.create_dataset("frac_x", data=crystals.frac_x)
        group.create_dataset("atomic_numbers", data=crystals.atomic_numbers)
        group.create_dataset("lattice", data=crystals.lattice)
        group.create_dataset("idx_start", data=crystals.idx_start)
        group.create_dataset("num_atoms", data=crystals.num_atoms)


def load_sample_results_from_hdf5(
    filename: str,
) -> tuple[SampleResult, AtomicNumberTable]:
    filepath = pathlib.Path(__file__).parent.resolve()
    with h5py.File(f"{filepath}/../../{filename}", "r") as file:
        sample_results = SampleResult(
            frac_x=file["crystals"]["frac_x"][:],
            atomic_numbers=file["crystals"]["atomic_numbers"][:],
            lattice=file["crystals"]["lattice"][:],
            num_atoms=file["crystals"]["num_atoms"][:],
            idx_start=file["crystals"]["idx_start"][:],
        )
    return sample_results


def get_crystal_indexes(sample_result: SampleResult, sample_idx: int):
    crystal_start_idx = sample_result.idx_start[sample_idx]
    num_atoms = sample_result.num_atoms[sample_idx]
    end_idx = crystal_start_idx + num_atoms
    return crystal_start_idx, end_idx


def get_one_crystal(sample_result: SampleResult, sample_idx: int):
    lattice = sample_result.lattice[sample_idx]

    crystal_start_idx, end_idx = get_crystal_indexes(sample_result, sample_idx)

    frac_x = sample_result.frac_x[crystal_start_idx:end_idx]
    atomic_numbers = sample_result.atomic_numbers[crystal_start_idx:end_idx]
    return lattice, frac_x, atomic_numbers
