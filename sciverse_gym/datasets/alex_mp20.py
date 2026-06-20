"""ALEX-MP-20 parquet dataset loader.

By Dr.Bin Cao (https://bin-cao.github.io/)

This file loads crystal structures and materials properties from the local
merged parquet database.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sciverse_gym.structures import StructureRecord, record_from_row


class AlexMP20Dataset:
    """Lazy row sampler for the local merged ALEX-MP-20 parquet file."""

    def __init__(
        self,
        data_path="data/alex-mp-20",
        split: str = "all",
        max_rows: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.path = self.data_path / "all.parquet"
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        self.frame = pd.read_parquet(self.path)
        self.split = split
        if split != "all":
            if "dataset_split" not in self.frame:
                raise ValueError(
                    f"split='{split}' requested, but {self.path} has no dataset_split column."
                )
            self.frame = self.frame[self.frame["dataset_split"] == split]
        if max_rows is not None:
            self.frame = self.frame.head(int(max_rows))
        if self.frame.empty:
            raise ValueError(f"Dataset is empty: {self.path}")

    def __len__(self):
        return len(self.frame)

    def get(self, index: int) -> StructureRecord:
        """Return one structure record by integer index."""
        row = self.frame.iloc[int(index) % len(self.frame)]
        return record_from_row(row)

    def sample(self, rng: np.random.Generator) -> StructureRecord:
        """Return a random structure record."""
        return self.get(int(rng.integers(0, len(self.frame))))
