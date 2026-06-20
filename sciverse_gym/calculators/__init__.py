"""Calculator backends for SciVerseGym.

By Dr.Bin Cao (https://bin-cao.github.io/)

This file exposes the unified property-evaluation interface and backends.
"""

from .base import BaseCalculator, CalculatorResult
from .factory import build_alex_mp20_physical_calculator
from .mattersim import MatterSimCalculatorFactory, mattersim_calculator_factory
from .mlff import (
    SUPPORTED_MLFF_MODELS,
    build_mlff_calculator,
    mlff_calculator_factory,
    mlff_energy,
    mlff_evaluate,
)
from .orb import ORBCalculatorFactory, orb_calculator_factory
from .physical import HullReferenceEntry, PhysicalCalculator
from .references import ReferenceEnergyDatabase, build_reference_database
from .sevennet import SevenNetCalculatorFactory, sevennet_calculator_factory
from .surrogate import SurrogateCalculator

__all__ = [
    "BaseCalculator",
    "CalculatorResult",
    "HullReferenceEntry",
    "MatterSimCalculatorFactory",
    "ORBCalculatorFactory",
    "PhysicalCalculator",
    "ReferenceEnergyDatabase",
    "SUPPORTED_MLFF_MODELS",
    "SevenNetCalculatorFactory",
    "SurrogateCalculator",
    "build_alex_mp20_physical_calculator",
    "build_mlff_calculator",
    "build_reference_database",
    "mattersim_calculator_factory",
    "mlff_calculator_factory",
    "mlff_energy",
    "mlff_evaluate",
    "orb_calculator_factory",
    "sevennet_calculator_factory",
]
