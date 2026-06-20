"""SciVerseGym package entry point.

By Dr.Bin Cao (https://bin-cao.github.io/)

This file registers crystal discovery environments with Gymnasium.
"""

from gymnasium import register

register(
    id="CrystalDiscovery-v0",
    entry_point="sciverse_gym.envs.crystal_discovery:CrystalDiscoveryEnv",
)

__all__ = ["CrystalDiscoveryEnv"]


def __getattr__(name):
    if name == "CrystalDiscoveryEnv":
        from sciverse_gym.envs.crystal_discovery import CrystalDiscoveryEnv

        return CrystalDiscoveryEnv
    raise AttributeError(name)
