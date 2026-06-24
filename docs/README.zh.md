<div align="center">

# SciVerseGym

### SVGym：面向晶体发现的 Gymnasium 环境

[![GitHub stars](https://img.shields.io/github/stars/Bin-Cao/SciVerseGym?style=social)](https://github.com/Bin-Cao/SciVerseGym/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Bin-Cao/SciVerseGym?style=social)](https://github.com/Bin-Cao/SciVerseGym/forks)
[![arXiv](https://img.shields.io/badge/arXiv-2606.22425-b31b1b.svg)](https://arxiv.org/abs/2606.22425)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-compatible-1f883d)](https://gymnasium.farama.org/)

[项目主页](https://bin-cao.github.io/SciVerseGym/) ·
[代码仓库](https://github.com/Bin-Cao/SciVerseGym) ·
[论文](https://arxiv.org/abs/2606.22425) ·
[手册](manual.html)

**语言：** [English](../README.md) | 中文 | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Deutsch](README.de.md)

</div>

---

SciVerseGym (SVGym) 是一个 Gymnasium 风格的晶体结构发现环境。智能体提交结构化的晶体编辑动作，环境返回标准的 Gymnasium 结果：

```python
obs, reward, terminated, truncated, info = env.step(action)
```

## 项目亮点

| 模块 | 能力 |
| --- | --- |
| 环境 | 用于 MLFF 支持的晶体发现任务的 `CrystalDiscovery-v0` |
| 动作 | 元素替换、晶格扰动、原子位移、空位和插入 |
| MLFF 后端 | 内置 SevenNet、MatterSim 和 ORB checkpoint |
| 数据集 | `data/alex-mp-20/` 下的本地 ALEX-MP-20 parquet 数据 |
| 基线 | 最小化贝叶斯优化与随机 rollout 示例 |
| 文档 | `docs/manual.html` 提供英文/中文双语完整手册 |

## 安装

```bash
python -m pip install -e ".[dev,mlff]"
```

## 快速开始

```python
import gymnasium as gym
import sciverse_gym

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
```

## 更多信息

项目主页：[https://bin-cao.github.io/SciVerseGym/](https://bin-cao.github.io/SciVerseGym/)

项目仓库：[https://github.com/Bin-Cao/SciVerseGym](https://github.com/Bin-Cao/SciVerseGym)

论文：[https://arxiv.org/abs/2606.22425](https://arxiv.org/abs/2606.22425)
