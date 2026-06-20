"""Shared calculator API.

By Dr.Bin Cao (https://bin-cao.github.io/)

This file defines the return type and interface for ML force-field, physical,
and custom property calculators.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ase import Atoms


@dataclass
class CalculatorResult:
    """Result returned after a structure is evaluated by the selected backend."""

    atoms: Atoms
    energy: float
    energy_above_hull: Optional[float]
    bulk_modulus: float = 0.0
    band_gap: float = 0.0
    formation_energy_per_atom: Optional[float] = None
    phonon_stable: Optional[bool] = None
    min_phonon_frequency: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    backend: str = "unknown"


class BaseCalculator:
    """Base class for MLFF, physical, and custom SciVerseGym calculators."""

    backend = "base"

    def evaluate(self, atoms: Atoms, reference: Optional[Dict[str, Any]] = None) -> CalculatorResult:
        """Evaluate a structure and return energy/property metadata."""
        raise NotImplementedError
