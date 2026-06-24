<div align="center">

# SciVerseGym

### SVGym: Eine Gymnasium-Umgebung für Kristallentdeckung

[![GitHub stars](https://img.shields.io/github/stars/Bin-Cao/SciVerseGym?style=social)](https://github.com/Bin-Cao/SciVerseGym/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Bin-Cao/SciVerseGym?style=social)](https://github.com/Bin-Cao/SciVerseGym/forks)
[![arXiv](https://img.shields.io/badge/arXiv-2606.22425-b31b1b.svg)](https://arxiv.org/abs/2606.22425)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-compatible-1f883d)](https://gymnasium.farama.org/)

[Website](https://bin-cao.github.io/SciVerseGym/) ·
[Repository](https://github.com/Bin-Cao/SciVerseGym) ·
[Paper](https://arxiv.org/abs/2606.22425) ·
[Manual](manual.html)

**Language:** [English](../README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | Deutsch

</div>

---

SciVerseGym (SVGym) ist eine Gymnasium-artige Umgebung für die Entdeckung von Kristallstrukturen. Agenten senden strukturierte Kristall-Editieraktionen, und die Umgebung gibt das standardisierte Gymnasium-Ergebnis zurück.

## Kernfunktionen

| Bereich | Funktion |
| --- | --- |
| Umgebung | `CrystalDiscovery-v0` für MLFF-gestützte Kristallentdeckung |
| Aktionen | Elementersetzung, Gitterstörung, Atomverschiebung, Leerstellen und Einfügung |
| MLFF | Mitgelieferte Checkpoints für SevenNet, MatterSim und ORB |
| Daten | Lokaler ALEX-MP-20 parquet-Datensatz unter `data/alex-mp-20/` |
| Baselines | Minimale Beispiele für Bayes'sche Optimierung und Random Rollout |
| Dokumentation | Ausführliches englisch/chinesisches Handbuch in `docs/manual.html` |

## Installation

```bash
python -m pip install -e ".[dev,mlff]"
```

## Befehle

```bash
python -m pytest
python -m sciverse_gym.benchmarks.bo_baselines.simple_bo --steps 5 --mlff-model sevennet
python -m sciverse_gym.benchmarks.rl_baselines.random_rollout --steps 5 --mlff-model sevennet
```

Weitere Informationen findest du im [manual](manual.html), im [repository](https://github.com/Bin-Cao/SciVerseGym) und im [paper](https://arxiv.org/abs/2606.22425).
