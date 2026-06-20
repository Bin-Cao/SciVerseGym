"""Random rollout baseline for CrystalDiscovery-v0.

By Dr.Bin Cao (https://bin-cao.github.io/)

This file provides a minimal executable baseline using an explicit MLFF backend
inside `gym.make(...)`; the default CLI backend is SevenNet.
"""

import argparse

import gymnasium as gym
import numpy as np

import sciverse_gym  # noqa: F401
from sciverse_gym.benchmarks.bo_baselines.simple_bo import candidate_actions


def main():
    """Run random valid structured actions in an MLFF-backed environment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--data-path", default="data/alex-mp-20")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mlff-model",
        default="sevennet",
        choices=("sevennet", "mattersim", "orb"),
        help="MLFF backend used by the environment.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = gym.make(
        "CrystalDiscovery-v0",
        data_path=args.data_path,
        mlff_model=args.mlff_model,
        mlff_relax=False,
        max_steps=args.steps,
    )
    obs, info = env.reset(seed=args.seed)
    total_reward = 0.0
    for _ in range(args.steps):
        actions = candidate_actions(obs)
        action = dict(actions[int(rng.integers(len(actions)))])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print({"total_reward": round(total_reward, 6), "last_info": info, "render": env.render()})


if __name__ == "__main__":
    main()
