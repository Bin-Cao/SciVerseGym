"""Reference database builders for force-field convex hull calculations."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from ase import Atoms
from ase.formula import Formula
from ase.optimize import BFGS

from sciverse_gym.structures import StructureRecord

from .physical import HullReferenceEntry


CalculatorFactory = Callable[[], object]


@dataclass(frozen=True)
class ReferenceEnergyDatabase:
    """Force-field energies needed to build composition hulls."""

    element_reference_energies: Dict[str, float]
    hull_reference_entries: List[HullReferenceEntry]

    def to_json(self, path):
        payload = {
            "element_reference_energies": self.element_reference_energies,
            "hull_reference_entries": [asdict(entry) for entry in self.hull_reference_entries],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def from_json(cls, path):
        payload = json.loads(Path(path).read_text())
        return cls(
            element_reference_energies={
                str(key): float(value)
                for key, value in payload["element_reference_energies"].items()
            },
            hull_reference_entries=[
                HullReferenceEntry(
                    composition=str(entry["composition"]),
                    formation_energy_per_atom=float(entry["formation_energy_per_atom"]),
                )
                for entry in payload["hull_reference_entries"]
            ],
        )

    def for_elements(self, elements: Iterable[str]):
        elements = set(elements)
        return ReferenceEnergyDatabase(
            element_reference_energies={
                element: energy
                for element, energy in self.element_reference_energies.items()
                if element in elements
            },
            hull_reference_entries=[
                entry
                for entry in self.hull_reference_entries
                if set(Formula(entry.composition).count()).issubset(elements)
            ],
        )


def build_reference_database(
    records,
    calculator_factory: CalculatorFactory,
    cache_path: Optional[str] = None,
    max_entries: Optional[int] = None,
    relax: bool = False,
    relax_fmax: float = 0.05,
    relax_steps: int = 100,
) -> ReferenceEnergyDatabase:
    """Evaluate a structure database and return same-force-field hull references.

    `records` may be an `AlexMP20Dataset`, any dataset with `get(index)`, or an
    iterable of `StructureRecord` objects.
    """
    if cache_path and Path(cache_path).exists():
        return ReferenceEnergyDatabase.from_json(cache_path)

    selected_records = _materialize_records(records, max_entries=max_entries)
    raw_entries = []
    element_reference_energies = {}

    for record in selected_records:
        atoms = record.atoms.copy()
        energy_per_atom = _energy_per_atom(
            atoms,
            calculator_factory=calculator_factory,
            relax=relax,
            relax_fmax=relax_fmax,
            relax_steps=relax_steps,
        )
        counts = dict(Formula(atoms.get_chemical_formula()).count())
        formula = atoms.get_chemical_formula(mode="hill")
        raw_entries.append((formula, counts, energy_per_atom))

        if len(counts) == 1:
            element = next(iter(counts))
            previous = element_reference_energies.get(element)
            if previous is None or energy_per_atom < previous:
                element_reference_energies[element] = energy_per_atom

    hull_entries = []
    for formula, counts, energy_per_atom in raw_entries:
        missing = sorted(set(counts) - set(element_reference_energies))
        if missing:
            continue
        reference = sum(
            count * element_reference_energies[element]
            for element, count in counts.items()
        ) / float(sum(counts.values()))
        hull_entries.append(
            HullReferenceEntry(
                composition=formula,
                formation_energy_per_atom=float(energy_per_atom - reference),
            )
        )

    database = ReferenceEnergyDatabase(
        element_reference_energies=element_reference_energies,
        hull_reference_entries=_deduplicate_lowest_entries(hull_entries),
    )
    if cache_path:
        database.to_json(cache_path)
    return database


def _materialize_records(records, max_entries: Optional[int]) -> List[StructureRecord]:
    limit = None if max_entries is None else int(max_entries)
    if hasattr(records, "get") and hasattr(records, "__len__"):
        count = len(records) if limit is None else min(len(records), limit)
        return [records.get(index) for index in range(count)]

    output = []
    for record in records:
        output.append(record)
        if limit is not None and len(output) >= limit:
            break
    return output


def _energy_per_atom(
    atoms: Atoms,
    calculator_factory: CalculatorFactory,
    relax: bool,
    relax_fmax: float,
    relax_steps: int,
) -> float:
    atoms.calc = calculator_factory()
    if relax:
        BFGS(atoms, logfile=None).run(fmax=relax_fmax, steps=relax_steps)
    return float(atoms.get_potential_energy() / max(len(atoms), 1))


def _deduplicate_lowest_entries(entries: Sequence[HullReferenceEntry]):
    lowest = {}
    for entry in entries:
        current = lowest.get(entry.composition)
        if current is None or entry.formation_energy_per_atom < current:
            lowest[entry.composition] = entry.formation_energy_per_atom
    return [
        HullReferenceEntry(composition=composition, formation_energy_per_atom=energy)
        for composition, energy in sorted(lowest.items())
        if np.isfinite(energy)
    ]
