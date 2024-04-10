import pathlib
import h5py
from diffusion.diffusion_loss import SampleResult

from diffusion.tools.atomic_number_table import AtomicNumberTable


def save_sample_results_to_hdf5(crystals: SampleResult, filename: str):
    with h5py.File(filename, "w") as file:
        group = file.create_group("crystals")
        group.create_dataset("x", data=crystals.x)
        group.create_dataset("atomic_numbers", data=crystals.atomic_numbers)
        group.create_dataset("lattice", data=crystals.lattice)
        group.create_dataset("idx_start", data=crystals.num_atoms)
        group.create_dataset("num_atoms", data=crystals.num_atoms)


def load_sample_results_from_hdf5(
    filename: str,
) -> tuple[SampleResult, AtomicNumberTable]:
    filepath = pathlib.Path(__file__).parent.resolve()
    with h5py.File(f"{filepath}/../../{filename}", "r") as file:
        sample_results = SampleResult(
            x=file["crystals"]["x"][:],
            atomic_numbers=file["crystals"]["atomic_numbers"][:],
            lattice=file["crystals"]["lattice"][:],
            num_atoms=file["crystals"]["num_atoms"][:],
            idx_start=file["crystals"]["idx_start"][:],
        )
    return sample_results
