"""Crystal edit actions.

By Dr.Bin Cao (https://bin-cao.github.io/)

This file implements structured crystal actions for replacement, lattice
perturbation, atom movement, vacancy creation, and insertion.
"""

from typing import Mapping

import numpy as np
from ase import Atoms

MAX_ATOMIC_NUMBER = 94


def apply_action(atoms: Atoms, action: Mapping) -> Atoms:
    """Apply one structured edit action and return a new structure."""
    edited = atoms.copy()
    action_type = int(action.get("action_type", 0))
    if action_type == 0:
        return replace_element(edited, action)
    if action_type == 1:
        return perturb_lattice(edited, action)
    if action_type == 2:
        return move_atom(edited, action)
    if action_type == 3:
        return remove_atom(edited, action)
    if action_type == 4:
        return insert_atom(edited, action)
    raise ValueError(f"Unknown action_type: {action_type}")


def replace_element(atoms: Atoms, action: Mapping) -> Atoms:
    """Replace the element at one site."""
    if len(atoms) == 0:
        return atoms
    site = _site(action, len(atoms))
    element = _atomic_number(action.get("element", atoms[site].number))
    atoms[site].number = element
    return atoms


def perturb_lattice(atoms: Atoms, action: Mapping) -> Atoms:
    """Perturb lattice vector lengths while preserving fractional positions."""
    delta = np.asarray(action.get("lattice_delta", [0.0, 0.0, 0.0]), dtype=float)
    delta = np.clip(delta, -0.15, 0.15)
    cell = atoms.cell.array.copy()
    for i in range(3):
        cell[i] *= max(0.5, 1.0 + float(delta[i]))
    atoms.set_cell(cell, scale_atoms=True)
    return atoms


def move_atom(atoms: Atoms, action: Mapping) -> Atoms:
    """Move one atom by a clipped Cartesian displacement."""
    if len(atoms) == 0:
        return atoms
    site = _site(action, len(atoms))
    delta = np.asarray(action.get("atom_delta", [0.0, 0.0, 0.0]), dtype=float)
    atoms.positions[site] += np.clip(delta, -0.25, 0.25)
    atoms.wrap()
    return atoms


def remove_atom(atoms: Atoms, action: Mapping) -> Atoms:
    """Remove one atom while keeping at least one site."""
    if len(atoms) <= 1:
        return atoms
    site = _site(action, len(atoms))
    del atoms[site]
    return atoms


def insert_atom(atoms: Atoms, action: Mapping) -> Atoms:
    """Insert one atom at a fractional coordinate."""
    element = _atomic_number(action.get("element", 1))
    frac = np.asarray(action.get("insert_position", [0.5, 0.5, 0.5]), dtype=float)
    frac = np.mod(frac, 1.0)
    position = frac @ atoms.cell.array
    atoms += Atoms(numbers=[element], positions=[position], cell=atoms.cell, pbc=True)
    atoms.wrap()
    return atoms


def _site(action: Mapping, natoms: int) -> int:
    return int(action.get("site", 0)) % max(natoms, 1)


def _atomic_number(value) -> int:
    return int(np.clip(int(value), 1, MAX_ATOMIC_NUMBER))
