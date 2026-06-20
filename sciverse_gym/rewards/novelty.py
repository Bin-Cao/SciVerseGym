"""Composition novelty reward terms.

By Dr.Bin Cao (https://bin-cao.github.io/)

This file rewards moving away from already-seen formulas.
"""

from ase import Atoms

from sciverse_gym.structures import canonical_formula


def composition_novelty_reward(atoms: Atoms, seen_formulas) -> float:
    """Reward formulas that have not been observed in the current episode."""
    formula = canonical_formula(atoms)
    return 0.25 if formula not in seen_formulas else 0.0
