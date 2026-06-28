<div align="center">

# SciVerseGym

### SVGym: A Gymnasium Environment for Crystal Discovery

[![GitHub stars](https://img.shields.io/github/stars/Bin-Cao/SciVerseGym?style=social)](https://github.com/Bin-Cao/SciVerseGym/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Bin-Cao/SciVerseGym?style=social)](https://github.com/Bin-Cao/SciVerseGym/forks)
[![arXiv](https://img.shields.io/badge/arXiv-2606.22425-b31b1b.svg)](https://arxiv.org/abs/2606.22425)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-compatible-1f883d)](https://gymnasium.farama.org/)
[![MLFF](https://img.shields.io/badge/MLFF-SevenNet%20%7C%20MatterSim%20%7C%20ORB-6f42c1)](#highlights)

[Website](https://bin-cao.github.io/SciVerseGym/) ·
[Repository](https://github.com/Bin-Cao/SciVerseGym) ·
[Paper](https://arxiv.org/abs/2606.22425) ·
[Manual](docs/manual.html)

**Language:** English | [中文](docs/README.zh.md) | [日本語](docs/README.ja.md) | [한국어](docs/README.ko.md) | [Español](docs/README.es.md) | [Deutsch](docs/README.de.md)

</div>

---
<img width="500" alt="Screenshot 2026-06-28 at 18 56 31" src="https://github.com/user-attachments/assets/18f81f82-d66a-4943-9d93-6e32a9f43488" />


SciVerseGym (SVGym) is a Gymnasium-style environment for crystal-structure discovery. Agents submit structured crystal-edit actions, and the environment returns the standard Gymnasium step tuple:

```python
obs, reward, terminated, truncated, info = env.step(action)
```

## Highlights

| Area | Capability |
| --- | --- |
| Environment | `CrystalDiscovery-v0` for MLFF-backed crystal discovery |
| Actions | Element replacement, lattice perturbation, atom displacement, vacancy, and insertion |
| MLFF backends | Packaged SevenNet, MatterSim, and ORB checkpoints |
| Dataset | Local ALEX-MP-20 parquet dataset under `data/alex-mp-20/` |
| Baselines | Minimal Bayesian optimization and random-rollout examples |
| Documentation | Full bilingual English/Chinese manual in `docs/manual.html` |

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

Open `docs/manual.html` in a browser for the full manual. It covers installation, all `gym.make(...)` parameters, action schemas, atomic numbers, BO/RL examples, reward calculation, formation energy, convex hull, phonons, datasets, and troubleshooting.

## Crystal Playground

Open `docs/crystal-playground.html` directly in a browser to try the bilingual English/Chinese SVGym Crystal Playground. It runs locally with no server or extra dependencies. Use the controls to generate many crystal presets, upload a CIF file with symmetry expansion into the full unit cell, adjust the lattice, add displacement or vacancies, run SVGym-style edit steps, inspect the local zoom view, relax the structure, and export the current observation as JSON.

## Citation

If you use SVGym (SciVerseGym), please cite:

```bibtex
@misc{cao2026svgymsciversegymenvironmentreinforcement,
      title={SVGym (SciVerseGym): An Environment for Reinforcement Learning and Bayesian Optimization in Crystal Discovery},
      author={Bin Cao},
      year={2026},
      eprint={2606.22425},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.22425},
}
```
