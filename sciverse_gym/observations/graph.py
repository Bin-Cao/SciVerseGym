"""Crystal graph observation builder.

By Dr.Bin Cao (https://bin-cao.github.io/)

This file converts ASE atoms to PyTorch Geometric graphs when available.
"""

from functools import lru_cache
import warnings

import numpy as np
import torch


@lru_cache(maxsize=1)
def _torch_geometric_data_cls():
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="An issue occurred while importing.*",
                category=UserWarning,
                module=r"torch_geometric\..*",
            )
            from torch_geometric.data import Data

        return Data
    except Exception:
        return None


def atoms_to_graph(atoms, cutoff: float = 5.0):
    """Build a radius graph from an ASE Atoms object."""
    numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    edge_index = _edge_index(atoms, cutoff=cutoff)
    data = {
        "x": torch.tensor(numbers[:, None], dtype=torch.float32),
        "pos": torch.tensor(positions, dtype=torch.float32),
        "cell": torch.tensor(np.asarray(atoms.cell.array), dtype=torch.float32),
        "edge_index": torch.tensor(edge_index, dtype=torch.long),
        "atomic_numbers": torch.tensor(numbers, dtype=torch.long),
    }
    data_cls = _torch_geometric_data_cls()
    if data_cls is None:
        return data
    return data_cls(**data)


def _edge_index(atoms, cutoff: float):
    if len(atoms) <= 1:
        return np.empty((2, 0), dtype=np.int64)
    distances = atoms.get_all_distances(mic=True)
    src, dst = np.where((distances > 0.0) & (distances <= cutoff))
    return np.vstack([src, dst]).astype(np.int64)
