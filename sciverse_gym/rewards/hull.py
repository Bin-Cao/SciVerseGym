"""Energy-above-hull reward terms.

By Dr.Bin Cao (https://bin-cao.github.io/)

This file defines stability rewards for low hull-energy structures.
"""


def hull_stability_reward(energy_above_hull: float, scale: float = 0.1) -> float:
    """Map lower energy above hull to higher reward.

    Returns zero when `energy_above_hull` is `None`. Otherwise the score is
    `1 / (1 + max(energy_above_hull, 0) / scale)`, with `scale=0.1` eV/atom by
    default.
    """
    if energy_above_hull is None:
        return 0.0
    scale = max(float(scale), 1e-8)
    return 1.0 / (1.0 + max(float(energy_above_hull), 0.0) / scale)
