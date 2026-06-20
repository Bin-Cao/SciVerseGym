"""High-level calculator factories for product use."""

from pathlib import Path
from typing import Optional

from sciverse_gym.datasets import AlexMP20Dataset

from .mlff import mlff_calculator_factory
from .physical import PhysicalCalculator
from .references import build_reference_database


def build_alex_mp20_physical_calculator(
    data_path="data/alex-mp-20",
    calculator_factory=None,
    model: str = "sevennet",
    reference_cache_path: Optional[str] = None,
    max_reference_entries: Optional[int] = None,
    relax: bool = True,
    relax_cell: bool = False,
    relax_references: bool = False,
    compute_phonons: bool = False,
    phonon_supercell=(2, 2, 2),
    phonon_mesh=(8, 8, 8),
) -> PhysicalCalculator:
    """Build an MLFF calculator with same-force-field FE/hull references.

    This is the recommended entry point when users need
    `formation_energy_per_atom` and `energy_above_hull` instead of only total
    MLFF energy. The function evaluates ALEX-MP-20 reference structures with
    the selected force field, derives elemental reference energies, builds
    formation-energy hull entries, caches them, and returns a
    `PhysicalCalculator` that can be passed as `calculator=...` to
    `gym.make(...)`.

    Args:
        data_path: ALEX-MP-20 parquet directory.
        calculator_factory: Optional custom ASE calculator factory. If omitted,
            `model` selects a packaged MLFF factory.
        model: Packaged MLFF backend used for references and candidates.
        reference_cache_path: JSON cache for elemental references and hull
            entries. Defaults to `<data_path>/<model>_reference_hull.json`.
        max_reference_entries: Optional cap on reference structures used to
            build the cache.
        relax: Whether candidate structures are relaxed during environment
            evaluation.
        relax_cell: Whether candidate relaxation may change the cell.
        relax_references: Whether reference structures are relaxed while
            building the cache.
        compute_phonons: Whether candidate evaluation also computes phonon
            stability.
        phonon_supercell: Supercell used by Phonopy if phonons are enabled.
        phonon_mesh: Mesh used by Phonopy if phonons are enabled.
    """
    if calculator_factory is None:
        calculator_factory = _calculator_factory_from_model(model)

    dataset = AlexMP20Dataset(data_path=data_path, split="all", max_rows=max_reference_entries)
    if reference_cache_path is None:
        reference_cache_path = str(Path(data_path) / f"{model}_reference_hull.json")

    reference_database = build_reference_database(
        dataset,
        calculator_factory=calculator_factory,
        cache_path=reference_cache_path,
        max_entries=max_reference_entries,
        relax=relax_references,
    )
    return PhysicalCalculator(
        calculator_factory=calculator_factory,
        element_reference_energies=reference_database.element_reference_energies,
        hull_reference_entries=reference_database.hull_reference_entries,
        relax=relax,
        relax_cell=relax_cell,
        compute_phonons=compute_phonons,
        phonon_supercell=phonon_supercell,
        phonon_mesh=phonon_mesh,
    )


def _calculator_factory_from_model(model: str):
    return mlff_calculator_factory(model)
