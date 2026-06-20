"""Multi-objective materials discovery reward.

By Dr.Bin Cao (https://bin-cao.github.io/)

This file combines stability, abundance, and novelty terms.
"""

import numpy as np

from sciverse_gym.calculators.base import CalculatorResult

from .hull import hull_stability_reward
from .novelty import composition_novelty_reward


def discovery_reward(result: CalculatorResult, seen_formulas) -> float:
    """Compute the default first-generation discovery reward.

    The scalar reward is:

        0.80 * stability + 0.10 * abundance_proxy + 0.10 * novelty

    `stability` is derived from same-force-field energy above hull. If hull
    references are unavailable and `energy_above_hull` is `None`, the stability
    term is zero rather than fabricated from total energy. `bulk_modulus` is
    reported in `info` when a backend provides it, but it does not participate
    in the current reward.
    """
    stability = hull_stability_reward(result.energy_above_hull)
    numbers = result.atoms.get_atomic_numbers()
    abundance_proxy = 1.0 / (1.0 + float(np.mean(numbers)) / 60.0) if len(numbers) else 0.0
    novelty = composition_novelty_reward(result.atoms, seen_formulas)
    return float(0.80 * stability + 0.10 * abundance_proxy + 0.10 * novelty)
