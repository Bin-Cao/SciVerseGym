"""Fast deterministic surrogate calculator.

By Dr.Bin Cao (https://bin-cao.github.io/)

This file provides an offline property evaluator for tests and baseline runs
when production ML force fields are unavailable.
"""

import math
from typing import Any, Dict, Optional

import numpy as np
from ase import Atoms

from .base import BaseCalculator, CalculatorResult


class SurrogateCalculator(BaseCalculator):
    """Cheap structure evaluator that mimics stability trends."""

    backend = "surrogate"

    def __init__(self, relaxation_scale: float = 0.02):
        self.relaxation_scale = float(relaxation_scale)

    def evaluate(self, atoms: Atoms, reference: Optional[Dict[str, Any]] = None) -> CalculatorResult:
        """Evaluate structure properties using geometry and composition features."""
        relaxed = atoms.copy()
        positions = relaxed.get_positions()
        if len(positions):
            center = positions.mean(axis=0)
            relaxed.set_positions(positions - self.relaxation_scale * (positions - center))

        numbers = relaxed.get_atomic_numbers()
        volume = max(float(relaxed.get_volume()), 1e-6)
        natoms = max(len(numbers), 1)
        mean_z = float(np.mean(numbers)) if len(numbers) else 0.0
        diversity = len(set(int(z) for z in numbers)) / natoms
        density_like = natoms / volume
        geometry_penalty = self._short_distance_penalty(relaxed)

        reference_hull = self._finite(reference, "energy_above_hull", default=0.05)
        reference_bulk = self._finite(reference, "ml_bulk_modulus", default=50.0)
        energy_above_hull = max(
            0.0,
            0.65 * reference_hull
            + 0.015 * abs(density_like - 0.08)
            + 0.0008 * diversity * mean_z
            + geometry_penalty,
        )
        energy = -0.04 * mean_z * natoms + 0.2 * energy_above_hull + geometry_penalty
        bulk_modulus = max(0.0, 0.85 * reference_bulk + 120.0 * density_like - 10.0 * diversity)
        band_gap = max(0.0, 3.0 / (1.0 + math.exp((mean_z - 22.0) / 8.0)) - 2.0 * diversity)

        return CalculatorResult(
            atoms=relaxed,
            energy=float(energy),
            energy_above_hull=float(energy_above_hull),
            bulk_modulus=float(bulk_modulus),
            band_gap=float(band_gap),
            metadata={"short_distance_penalty": geometry_penalty, "density_like": density_like},
            backend=self.backend,
        )

    @staticmethod
    def _finite(reference: Optional[Dict[str, Any]], key: str, default: float) -> float:
        if not reference:
            return default
        value = reference.get(key, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if np.isfinite(value) else default

    @staticmethod
    def _short_distance_penalty(atoms: Atoms) -> float:
        if len(atoms) < 2:
            return 0.0
        distances = atoms.get_all_distances(mic=True)
        distances[distances == 0.0] = np.inf
        min_distance = float(np.min(distances))
        return max(0.0, 1.0 - min_distance) * 0.25
