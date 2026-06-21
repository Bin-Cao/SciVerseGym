# SVGym

SciVerseGym ([SVGym](https://bin-cao.github.io/SciVerseGym/)) is a Gymnasium-style environment for crystal-structure discovery. Agents submit structured crystal-edit actions, and the environment returns the standard:

```python
obs, reward, terminated, truncated, info = env.step(action)
```

## Highlights

- [ENV] `CrystalDiscovery-v0` for MLFF-backed crystal discovery.
- [ACT] Element replacement, lattice perturbation, atom displacement, vacancy, and insertion.
- [MLFF] Packaged SevenNet, MatterSim, and ORB checkpoints.
- [DATA] Local ALEX-MP-20 parquet dataset under `data/alex-mp-20/`.
- [BO/RL] Minimal Bayesian optimization and random-rollout examples.
- [DOC] Complete bilingual English/Chinese manual: `docs/manual_zh.html`.

## Install

```bash
python -m pip install -e ".[dev,mlff]"
```

## Quick Start

```python
import gymnasium as gym
import sciverse_gym  # registers CrystalDiscovery-v0

env = gym.make(
    "CrystalDiscovery-v0",
    data_path="data/alex-mp-20",
    mlff_model="sevennet",
    mlff_relax=False,
    max_steps=5,
    max_dataset_rows=20,
)

obs, info = env.reset(seed=0, options={"index": 0})
action = {"action_type": 0, "site": 0, "element": 28}
obs, reward, terminated, truncated, info = env.step(action)
print(info["formula"], float(obs["energy"]), reward)
```

## Commands

```bash
python -m pytest
python -m sciverse_gym.benchmarks.bo_baselines.simple_bo --steps 5 --mlff-model sevennet
python -m sciverse_gym.benchmarks.rl_baselines.random_rollout --steps 5 --mlff-model sevennet
```

## Documentation

Open `docs/manual.html` in a browser for the full bilingual manual. It covers installation, all `gym.make(...)` parameters, action schemas, atomic numbers, BO/RL training examples, reward calculation, formation energy, convex hull, phonons, datasets, and troubleshooting.
