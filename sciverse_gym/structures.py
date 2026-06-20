"""Crystal structure conversion utilities.

By Dr.Bin Cao (https://bin-cao.github.io/)

This file converts parquet rows and edited atom states into ASE structures.
"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from ase import Atoms


@dataclass(frozen=True)
class StructureRecord:
    """Single crystal entry loaded from a materials dataset."""

    atoms: Atoms
    material_id: str
    metadata: Dict[str, Any]


def atoms_from_row(row) -> Atoms:
    """Create an ASE Atoms object from an ALEX-MP-20 parquet row."""
    positions = _nested_array(row["positions"], dtype=float)
    cell = _nested_array(row["cell"], dtype=float)
    numbers = np.asarray(row["atomic_numbers"], dtype=int)
    return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)


def _nested_array(value, dtype):
    """Convert Arrow/Pandas nested object arrays into a dense numeric array."""
    array = np.asarray(value)
    if array.dtype == object:
        return np.stack([np.asarray(item, dtype=dtype) for item in array]).astype(dtype)
    return np.asarray(value, dtype=dtype)


def record_from_row(row) -> StructureRecord:
    """Create a structure record with metadata preserved."""
    metadata = {
        "space_group": row.get("space_group"),
        "chemical_system": row.get("chemical_system"),
        "energy_above_hull": float(row.get("energy_above_hull", np.nan)),
        "dft_band_gap": float(row.get("dft_band_gap", np.nan)),
        "dft_bulk_modulus": float(row.get("dft_bulk_modulus", np.nan)),
        "dft_mag_density": float(row.get("dft_mag_density", np.nan)),
        "hhi_score": float(row.get("hhi_score", np.nan)),
        "ml_bulk_modulus": float(row.get("ml_bulk_modulus", np.nan)),
    }
    return StructureRecord(
        atoms=atoms_from_row(row),
        material_id=str(row.get("material_id", "")),
        metadata=metadata,
    )


def canonical_formula(atoms: Atoms) -> str:
    """Return a stable formula string for novelty tracking."""
    return atoms.get_chemical_formula(mode="hill")
