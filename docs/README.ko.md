<div align="center">

# SciVerseGym

### SVGym: 결정 발견을 위한 Gymnasium 환경

[![GitHub stars](https://img.shields.io/github/stars/Bin-Cao/SciVerseGym?style=social)](https://github.com/Bin-Cao/SciVerseGym/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Bin-Cao/SciVerseGym?style=social)](https://github.com/Bin-Cao/SciVerseGym/forks)
[![arXiv](https://img.shields.io/badge/arXiv-2606.22425-b31b1b.svg)](https://arxiv.org/abs/2606.22425)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-compatible-1f883d)](https://gymnasium.farama.org/)

[Website](https://bin-cao.github.io/SciVerseGym/) ·
[Repository](https://github.com/Bin-Cao/SciVerseGym) ·
[Paper](https://arxiv.org/abs/2606.22425) ·
[Manual](manual.html)

**Language:** [English](../README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | 한국어 | [Español](README.es.md) | [Deutsch](README.de.md)

</div>

---

SciVerseGym (SVGym)은 결정 구조 발견을 위한 Gymnasium 스타일 환경입니다. 에이전트는 구조화된 결정 편집 액션을 제출하고, 환경은 표준 Gymnasium 결과를 반환합니다.

## 주요 기능

| 영역 | 기능 |
| --- | --- |
| 환경 | MLFF 기반 결정 발견을 위한 `CrystalDiscovery-v0` |
| 액션 | 원소 치환, 격자 섭동, 원자 변위, 공공, 삽입 |
| MLFF | SevenNet, MatterSim, ORB checkpoint 포함 |
| 데이터 | `data/alex-mp-20/` 아래의 ALEX-MP-20 parquet 데이터셋 |
| 기준선 | 간단한 베이지안 최적화 및 랜덤 rollout 예제 |
| 문서 | `docs/manual.html`의 영어/중국어 상세 매뉴얼 |

## 설치

```bash
python -m pip install -e ".[dev,mlff]"
```

## 실행 명령

```bash
python -m pytest
python -m sciverse_gym.benchmarks.bo_baselines.simple_bo --steps 5 --mlff-model sevennet
python -m sciverse_gym.benchmarks.rl_baselines.random_rollout --steps 5 --mlff-model sevennet
```

자세한 내용은 [manual](manual.html), [repository](https://github.com/Bin-Cao/SciVerseGym), [paper](https://arxiv.org/abs/2606.22425)를 확인하세요.
