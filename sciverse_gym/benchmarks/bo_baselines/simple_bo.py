"""Minimal Bayesian optimization loop for CrystalDiscovery-v0.

By Dr.Bin Cao (https://bin-cao.github.io/)

This example keeps the BO machinery intentionally small. It creates the
environment with `mlff_model="sevennet"` by default, chooses an action, submits
it with `env.step(action)`, reads the new observation/state, and updates the
optimizer with the observed reward.
"""

import argparse
from typing import Dict, List, Sequence

import gymnasium as gym
import numpy as np

import sciverse_gym  # noqa: F401


def candidate_actions(obs: Dict) -> List[Dict]:
    """Build structured edit actions that BO can submit for the current state."""
    natoms = len(obs["structure"])
    site_count = min(natoms, 4)
    actions: List[Dict] = []

    for site in range(site_count):
        for element in (3, 6, 8, 14, 26, 28, 32):
            actions.append({"action_type": 0, "site": site, "element": element})

    for delta in (-0.06, -0.03, 0.03, 0.06):
        actions.append({"action_type": 1, "lattice_delta": [delta, 0.0, 0.0]})
        actions.append({"action_type": 1, "lattice_delta": [0.0, delta, 0.0]})
        actions.append({"action_type": 1, "lattice_delta": [0.0, 0.0, delta]})

    for site in range(min(natoms, 2)):
        for delta in (-0.08, 0.08):
            actions.append({"action_type": 2, "site": site, "atom_delta": [delta, 0.0, 0.0]})
            actions.append({"action_type": 2, "site": site, "atom_delta": [0.0, delta, 0.0]})
            actions.append({"action_type": 2, "site": site, "atom_delta": [0.0, 0.0, delta]})

    if natoms > 1:
        for site in range(site_count):
            actions.append({"action_type": 3, "site": site})

    return actions


def featurize(obs: Dict, action: Dict) -> np.ndarray:
    """Convert one `(state, action)` pair into numeric GP input features."""
    natoms = max(len(obs["structure"]), 1)
    action_type = int(action.get("action_type", 0))
    one_hot = np.zeros(5, dtype=float)
    one_hot[action_type] = 1.0

    lattice_delta = np.asarray(action.get("lattice_delta", [0.0, 0.0, 0.0]), dtype=float)
    atom_delta = np.asarray(action.get("atom_delta", [0.0, 0.0, 0.0]), dtype=float)
    insert_position = np.asarray(action.get("insert_position", [0.5, 0.5, 0.5]), dtype=float)

    return np.concatenate(
        [
            one_hot,
            np.asarray(
                [
                    float(obs["energy"]),
                    float(obs["step"]) / max(float(obs["step"] + obs["budget_left"]), 1.0),
                    float(natoms) / 64.0,
                    float(action.get("site", 0)) / float(natoms),
                    float(action.get("element", 0)) / 94.0,
                ]
            ),
            lattice_delta / 0.15,
            atom_delta / 0.25,
            insert_position,
        ]
    )


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float) -> np.ndarray:
    """Return the RBF covariance matrix used by BO's tiny GP model."""
    diff = x1[:, None, :] - x2[None, :, :]
    sqdist = np.sum(diff * diff, axis=2)
    return np.exp(-0.5 * sqdist / (length_scale * length_scale))


def choose_action(
    obs: Dict,
    actions: Sequence[Dict],
    x_train: Sequence[np.ndarray],
    y_train: Sequence[float],
    rng: np.random.Generator,
    exploration: float,
) -> Dict:
    """Choose the next action with Gaussian-process UCB over candidate actions."""
    if len(y_train) < 2:
        return dict(actions[int(rng.integers(len(actions)))])

    x_obs = np.vstack(x_train)
    y_obs = np.asarray(y_train, dtype=float)
    y_mean = float(np.mean(y_obs))
    y_std = float(np.std(y_obs) + 1e-8)
    y_scaled = (y_obs - y_mean) / y_std

    x_candidates = np.vstack([featurize(obs, action) for action in actions])
    kernel = rbf_kernel(x_obs, x_obs, length_scale=1.5)
    kernel += np.eye(len(x_obs)) * 1e-6
    cross_kernel = rbf_kernel(x_obs, x_candidates, length_scale=1.5)

    chol = np.linalg.cholesky(kernel)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_scaled))
    mean = cross_kernel.T @ alpha
    variance = 1.0 - np.sum(np.linalg.solve(chol, cross_kernel) ** 2, axis=0)
    std = np.sqrt(np.maximum(variance, 1e-9))
    score = mean + exploration * std
    return dict(actions[int(np.argmax(score))])


def main():
    """Run a minimal MLFF-backed BO loop from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-path", default="data/alex-mp-20")
    parser.add_argument("--max-dataset-rows", type=int, default=50)
    parser.add_argument("--exploration", type=float, default=1.5)
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
        max_dataset_rows=args.max_dataset_rows,
    )
    obs, info = env.reset(seed=args.seed)

    x_train: List[np.ndarray] = []
    y_train: List[float] = []
    best_reward = -np.inf
    best_info = None

    print("initial_state", {"formula": info["formula"], "energy": round(float(obs["energy"]), 6)})
    for _ in range(args.steps):
        actions = candidate_actions(obs)
        action = choose_action(obs, actions, x_train, y_train, rng, args.exploration)

        previous_obs = obs
        obs, reward, terminated, truncated, info = env.step(action)

        x_train.append(featurize(previous_obs, action))
        y_train.append(float(reward))
        if reward > best_reward:
            best_reward = float(reward)
            best_info = dict(info)

        print(
            {
                "submitted_action": action,
                "reward": round(float(reward), 6),
                "new_state": {
                    "formula": info["formula"],
                    "energy": round(float(obs["energy"]), 6),
                    "step": obs["step"],
                    "budget_left": obs["budget_left"],
                },
            }
        )

        if terminated or truncated:
            break

    print({"best_reward": round(best_reward, 6), "best_info": best_info, "render": env.render()})


if __name__ == "__main__":
    main()
