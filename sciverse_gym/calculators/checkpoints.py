"""Packaged MLFF checkpoint paths."""

from importlib import resources
from pathlib import Path


CHECKPOINT_FILES = {
    "mattersim": "mattersim-v1.0.0-1M.pth",
    "orb": "orb-v3-conservative-inf-omat-20250404.ckpt",
    "sevennet": "sevennet-0-11jul2024.pth",
}


def checkpoint_path(model: str) -> Path:
    """Return the packaged checkpoint path for a built-in MLFF model."""
    normalized = model.lower().replace("-", "").replace("_", "")
    if normalized not in CHECKPOINT_FILES:
        supported = ", ".join(sorted(CHECKPOINT_FILES))
        raise ValueError(f"Unsupported MLFF checkpoint '{model}'. Supported: {supported}.")

    checkpoint = resources.files("sciverse_gym").joinpath(
        "mlff_checkpoints", CHECKPOINT_FILES[normalized]
    )
    return Path(str(checkpoint))
