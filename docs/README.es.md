<div align="center">

# SciVerseGym

### SVGym: Un entorno Gymnasium para descubrimiento cristalino

[![GitHub stars](https://img.shields.io/github/stars/Bin-Cao/SciVerseGym?style=social)](https://github.com/Bin-Cao/SciVerseGym/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Bin-Cao/SciVerseGym?style=social)](https://github.com/Bin-Cao/SciVerseGym/forks)
[![arXiv](https://img.shields.io/badge/arXiv-2606.22425-b31b1b.svg)](https://arxiv.org/abs/2606.22425)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-compatible-1f883d)](https://gymnasium.farama.org/)

[Website](https://bin-cao.github.io/SciVerseGym/) ·
[Repository](https://github.com/Bin-Cao/SciVerseGym) ·
[Paper](https://arxiv.org/abs/2606.22425) ·
[Manual](manual.html)

**Language:** [English](../README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | Español | [Deutsch](README.de.md)

</div>

---

SciVerseGym (SVGym) es un entorno de estilo Gymnasium para el descubrimiento de estructuras cristalinas. Los agentes envían acciones estructuradas de edición cristalina y el entorno devuelve la tupla estándar de Gymnasium.

## Puntos clave

| Área | Capacidad |
| --- | --- |
| Entorno | `CrystalDiscovery-v0` para descubrimiento cristalino con MLFF |
| Acciones | Reemplazo de elementos, perturbación de red, desplazamiento atómico, vacantes e inserción |
| MLFF | Checkpoints incluidos para SevenNet, MatterSim y ORB |
| Datos | Dataset local ALEX-MP-20 parquet en `data/alex-mp-20/` |
| Baselines | Ejemplos mínimos de optimización bayesiana y random rollout |
| Documentación | Manual detallado inglés/chino en `docs/manual.html` |

## Instalación

```bash
python -m pip install -e ".[dev,mlff]"
```

## Comandos

```bash
python -m pytest
python -m sciverse_gym.benchmarks.bo_baselines.simple_bo --steps 5 --mlff-model sevennet
python -m sciverse_gym.benchmarks.rl_baselines.random_rollout --steps 5 --mlff-model sevennet
```

## Crystal Playground

Abre `docs/crystal-playground.html` directamente en el navegador para usar el Crystal Playground bilingüe inglés/chino en local. No necesita servidor ni dependencias adicionales. Permite generar muchos preajustes cristalinos, subir un archivo CIF con expansión de simetría a la celda unidad completa, ajustar red y desplazamientos atómicos, añadir vacantes, ejecutar pasos de edición estilo SVGym, inspeccionar una vista local ampliada, relajar la estructura y exportar la observación como JSON.

Consulta el [manual](manual.html), el [repositorio](https://github.com/Bin-Cao/SciVerseGym) y el [artículo](https://arxiv.org/abs/2606.22425) para más detalles.
