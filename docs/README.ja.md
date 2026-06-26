<div align="center">

# SciVerseGym

### SVGym: 結晶探索のための Gymnasium 環境

[![GitHub stars](https://img.shields.io/github/stars/Bin-Cao/SciVerseGym?style=social)](https://github.com/Bin-Cao/SciVerseGym/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Bin-Cao/SciVerseGym?style=social)](https://github.com/Bin-Cao/SciVerseGym/forks)
[![arXiv](https://img.shields.io/badge/arXiv-2606.22425-b31b1b.svg)](https://arxiv.org/abs/2606.22425)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-compatible-1f883d)](https://gymnasium.farama.org/)

[Website](https://bin-cao.github.io/SciVerseGym/) ·
[Repository](https://github.com/Bin-Cao/SciVerseGym) ·
[Paper](https://arxiv.org/abs/2606.22425) ·
[Manual](manual.html)

**Language:** [English](../README.md) | [中文](README.zh.md) | 日本語 | [한국어](README.ko.md) | [Español](README.es.md) | [Deutsch](README.de.md)

</div>

---

SciVerseGym (SVGym) は、結晶構造探索のための Gymnasium 形式の環境です。エージェントは構造化された結晶編集アクションを送信し、環境は標準的な Gymnasium の結果を返します。

## 主な特徴

| 項目 | 内容 |
| --- | --- |
| 環境 | MLFF に基づく結晶探索用の `CrystalDiscovery-v0` |
| アクション | 元素置換、格子摂動、原子変位、空孔、挿入 |
| MLFF | SevenNet、MatterSim、ORB の checkpoint を同梱 |
| データ | `data/alex-mp-20/` 内の ALEX-MP-20 parquet データセット |
| ベースライン | ベイズ最適化とランダム rollout の最小実装 |
| ドキュメント | `docs/manual.html` に英語/中国語の詳細マニュアル |

## インストール

```bash
python -m pip install -e ".[dev,mlff]"
```

## 実行例

```bash
python -m pytest
python -m sciverse_gym.benchmarks.bo_baselines.simple_bo --steps 5 --mlff-model sevennet
python -m sciverse_gym.benchmarks.rl_baselines.random_rollout --steps 5 --mlff-model sevennet
```

## Crystal Playground

`docs/crystal-playground.html` をブラウザで直接開くと、英語/中国語の双方向 Crystal Playground をローカルで実行できます。サーバーや追加依存関係は不要です。多様な結晶プリセットの生成、CIF ファイルのアップロードと対称操作による完全単位胞への展開、格子と原子変位の調整、空孔の追加、SVGym 形式の編集ステップ、局所拡大表示、構造緩和、観測 JSON のエクスポートを試せます。

詳細は [manual](manual.html)、[repository](https://github.com/Bin-Cao/SciVerseGym)、[paper](https://arxiv.org/abs/2606.22425) を参照してください。
