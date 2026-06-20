"""Production physical-property calculator built on ASE-compatible force fields."""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Union

import numpy as np
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.formula import Formula
from ase.optimize import BFGS
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from scipy.optimize import linprog

from .base import BaseCalculator, CalculatorResult


CalculatorFactory = Callable[[], object]


@dataclass(frozen=True)
class HullReferenceEntry:
    """Reference entry for a formation-energy phase diagram."""

    composition: str
    formation_energy_per_atom: float


class PhysicalCalculator(BaseCalculator):
    """Evaluate structures with a real ASE calculator and optional phonons.

    This backend does not invent thermodynamic quantities. Formation energies
    require elemental reference chemical potentials from the same force field,
    and energy above hull requires reference phase-diagram entries in the same
    energy convention. A force field directly provides total energy and forces;
    formation energy and convex-hull quantities are derived from same-force-field
    reference data.
    """

    backend = "physical"

    def __init__(
        self,
        ase_calculator: Optional[object] = None,
        calculator_factory: Optional[CalculatorFactory] = None,
        element_reference_energies: Optional[Dict[str, float]] = None,
        hull_reference_entries: Optional[Iterable[HullReferenceEntry]] = None,
        relax: bool = True,
        relax_cell: bool = False,
        relax_fmax: float = 0.05,
        relax_steps: int = 200,
        compute_phonons: bool = False,
        phonon_supercell: Union[int, Sequence[int], Sequence[Sequence[int]]] = (2, 2, 2),
        phonon_displacement: float = 0.01,
        phonon_mesh: Union[int, Sequence[int]] = (8, 8, 8),
        phonon_stability_tolerance: float = -0.1,
    ):
        if ase_calculator is None and calculator_factory is None:
            raise ValueError("PhysicalCalculator requires an ASE calculator or calculator_factory.")
        self.ase_calculator = ase_calculator
        self.calculator_factory = calculator_factory
        self.element_reference_energies = dict(element_reference_energies or {})
        self.hull_reference_entries = list(hull_reference_entries or [])
        self.relax = bool(relax)
        self.relax_cell = bool(relax_cell)
        self.relax_fmax = float(relax_fmax)
        self.relax_steps = int(relax_steps)
        self.compute_phonons = bool(compute_phonons)
        self.phonon_supercell = phonon_supercell
        self.phonon_displacement = float(phonon_displacement)
        self.phonon_mesh = phonon_mesh
        self.phonon_stability_tolerance = float(phonon_stability_tolerance)

    def evaluate(self, atoms: Atoms, reference: Optional[Dict] = None) -> CalculatorResult:
        """Relax and evaluate a crystal using the configured force-field API."""
        del reference
        relaxed = atoms.copy()
        relaxed.calc = self._new_calculator()

        if self.relax:
            optimizable = FrechetCellFilter(relaxed) if self.relax_cell else relaxed
            BFGS(optimizable, logfile=None).run(fmax=self.relax_fmax, steps=self.relax_steps)

        energy = float(relaxed.get_potential_energy())
        formation = self._formation_energy_per_atom(relaxed, energy)
        energy_above_hull = self._energy_above_hull(relaxed, formation)

        phonon_stable = None
        min_phonon_frequency = None
        if self.compute_phonons:
            min_phonon_frequency = self._min_phonon_frequency(relaxed)
            phonon_stable = bool(min_phonon_frequency >= self.phonon_stability_tolerance)

        return CalculatorResult(
            atoms=relaxed,
            energy=energy,
            energy_above_hull=energy_above_hull,
            formation_energy_per_atom=formation,
            phonon_stable=phonon_stable,
            min_phonon_frequency=min_phonon_frequency,
            metadata={
                "energy_per_atom": energy / max(len(relaxed), 1),
                "relaxed": self.relax,
                "cell_relaxed": self.relax and self.relax_cell,
                "phonons_computed": self.compute_phonons,
            },
            backend=self.backend,
        )

    def _new_calculator(self):
        if self.calculator_factory is not None:
            return self.calculator_factory()
        return self.ase_calculator

    def _formation_energy_per_atom(self, atoms: Atoms, total_energy: float) -> Optional[float]:
        symbols = atoms.get_chemical_symbols()
        missing = sorted(set(symbols) - set(self.element_reference_energies))
        if missing:
            return None
        reference_energy = sum(self.element_reference_energies[symbol] for symbol in symbols)
        return float((total_energy - reference_energy) / max(len(atoms), 1))

    def _energy_above_hull(self, atoms: Atoms, formation_energy_per_atom: Optional[float]):
        if formation_energy_per_atom is None or not self.hull_reference_entries:
            return None

        current_counts = dict(Formula(atoms.get_chemical_formula()).count())
        elements = sorted(current_counts)
        target = self._composition_vector(current_counts, elements)

        reference_vectors = []
        reference_energies = []
        for entry in self.hull_reference_entries:
            counts = dict(Formula(entry.composition).count())
            if set(counts).issubset(elements):
                reference_vectors.append(self._composition_vector(counts, elements))
                reference_energies.append(float(entry.formation_energy_per_atom))

        for index in range(len(elements)):
            vector = np.zeros(len(elements), dtype=float)
            vector[index] = 1.0
            reference_vectors.append(vector)
            reference_energies.append(0.0)

        result = linprog(
            c=np.asarray(reference_energies, dtype=float),
            A_eq=np.vstack(reference_vectors).T,
            b_eq=target,
            bounds=[(0.0, None)] * len(reference_energies),
            method="highs",
        )
        if not result.success:
            return None
        hull_energy = float(result.fun)
        return max(0.0, float(formation_energy_per_atom) - hull_energy)

    def _min_phonon_frequency(self, atoms: Atoms) -> float:
        unitcell = PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            cell=atoms.cell.array,
            scaled_positions=atoms.get_scaled_positions(),
        )
        phonon = Phonopy(
            unitcell,
            supercell_matrix=self._supercell_matrix(self.phonon_supercell),
            log_level=0,
        )
        phonon.generate_displacements(distance=self.phonon_displacement)
        forces = []
        for supercell in phonon.supercells_with_displacements:
            displaced = Atoms(
                symbols=supercell.symbols,
                cell=supercell.cell,
                scaled_positions=supercell.scaled_positions,
                pbc=True,
            )
            displaced.calc = self._new_calculator()
            forces.append(displaced.get_forces())

        phonon.forces = forces
        phonon.produce_force_constants()
        phonon.run_mesh(self.phonon_mesh, is_gamma_center=True)
        mesh = phonon.get_mesh_dict()
        return float(np.min(mesh["frequencies"]))

    @staticmethod
    def _supercell_matrix(supercell):
        array = np.asarray(supercell, dtype=int)
        if array.ndim == 0:
            return np.eye(3, dtype=int) * int(array)
        if array.shape == (3,):
            return np.diag(array)
        if array.shape == (3, 3):
            return array
        raise ValueError("phonon_supercell must be an int, length-3 sequence, or 3x3 matrix.")

    @staticmethod
    def _composition_vector(counts: Dict[str, int], elements: Sequence[str]):
        total = float(sum(counts.values()))
        return np.asarray([float(counts.get(element, 0)) / total for element in elements])
