"""Built-in ML force-field entry points used by environment examples."""

from typing import Optional

from ase import Atoms

from .base import CalculatorResult
from .mattersim import MatterSimCalculatorFactory
from .orb import ORBCalculatorFactory
from .physical import PhysicalCalculator
from .sevennet import SevenNetCalculatorFactory


SUPPORTED_MLFF_MODELS = ("mattersim", "orb", "sevennet")


def mlff_calculator_factory(model: str = "sevennet", **kwargs):
    """Return an ASE calculator factory for a packaged MLFF backend."""
    normalized = _normalize_model(model)
    if normalized == "mattersim":
        return MatterSimCalculatorFactory(**kwargs)
    if normalized == "orb":
        return ORBCalculatorFactory(**kwargs)
    if normalized == "sevennet":
        return SevenNetCalculatorFactory(**kwargs)
    supported = ", ".join(SUPPORTED_MLFF_MODELS)
    raise ValueError(f"Unsupported built-in MLFF model '{model}'. Supported: {supported}.")


def build_mlff_calculator(
    model: str = "sevennet",
    relax: bool = False,
    relax_cell: bool = False,
    compute_phonons: bool = False,
    **factory_kwargs,
) -> PhysicalCalculator:
    """Build a `PhysicalCalculator` from a packaged MLFF checkpoint."""
    return PhysicalCalculator(
        calculator_factory=mlff_calculator_factory(model, **factory_kwargs),
        relax=relax,
        relax_cell=relax_cell,
        compute_phonons=compute_phonons,
    )


def mlff_evaluate(
    atoms: Atoms,
    model: str = "sevennet",
    calculator: Optional[PhysicalCalculator] = None,
    **kwargs,
) -> CalculatorResult:
    """Evaluate an ASE `Atoms` object with a built-in packaged MLFF."""
    evaluator = calculator or build_mlff_calculator(model=model, **kwargs)
    return evaluator.evaluate(atoms)


def mlff_energy(
    atoms: Atoms,
    model: str = "sevennet",
    per_atom: bool = False,
    **kwargs,
) -> float:
    """Return the total or per-atom MLFF energy for an ASE `Atoms` object."""
    result = mlff_evaluate(atoms, model=model, **kwargs)
    if per_atom:
        return float(result.energy / max(len(result.atoms), 1))
    return float(result.energy)


def _normalize_model(model: str) -> str:
    """Normalize user-facing MLFF model aliases to canonical names."""
    normalized = model.lower().replace("-", "").replace("_", "")
    aliases = {
        "mattersim": "mattersim",
        "orb": "orb",
        "orbv3": "orb",
        "sevennet": "sevennet",
        "sevenn": "sevennet",
        "7net": "sevennet",
    }
    return aliases.get(normalized, normalized)
