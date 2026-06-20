"""Lightweight XRD-like fingerprint helper.

By Dr.Bin Cao (https://bin-cao.github.io/)

This file builds a deterministic diffraction-inspired vector without external
simulation dependencies.
"""

import numpy as np


def xrd_fingerprint(atoms, bins: int = 64):
    """Return a normalized pair-distance histogram as an XRD-like feature."""
    if len(atoms) < 2:
        return np.zeros(bins, dtype=np.float32)
    distances = atoms.get_all_distances(mic=True)
    values = distances[np.triu_indices(len(atoms), k=1)]
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 10.0))
    total = hist.sum()
    return (hist / total if total else hist).astype(np.float32)
