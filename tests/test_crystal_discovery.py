"""Tests for CrystalDiscovery-v0.

By Dr.Bin Cao (https://bin-cao.github.io/)

This file verifies registration, reset, action dynamics, and truncation.
"""

import gymnasium as gym
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones

import sciverse_gym  # noqa: F401
from sciverse_gym.calculators import (
    SUPPORTED_MLFF_MODELS,
    CalculatorResult,
    PhysicalCalculator,
    build_mlff_calculator,
    build_reference_database,
)
from sciverse_gym.calculators.checkpoints import checkpoint_path
from sciverse_gym.datasets import AlexMP20Dataset
from sciverse_gym.rewards import discovery_reward
from sciverse_gym.structures import StructureRecord


def test_env_reset_and_step():
    env = gym.make("CrystalDiscovery-v0", max_steps=3, max_dataset_rows=20)
    obs, info = env.reset(seed=123, options={"index": 0})
    assert obs["step"] == 0
    assert obs["budget_left"] == 3
    assert "graph" in obs
    assert info["backend"] == "surrogate"

    obs, reward, terminated, truncated, info = env.step(
        {"action_type": 0, "site": 0, "element": 28}
    )
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    assert obs["step"] == 1
    assert info["formula"]


def test_env_truncates_at_budget():
    env = gym.make("CrystalDiscovery-v0", max_steps=1, max_dataset_rows=10)
    env.reset(seed=1)
    _, _, _, truncated, _ = env.step({"action_type": 1, "lattice_delta": [0.01, 0.0, 0.0]})
    assert truncated


def test_discovery_reward_ignores_bulk_modulus():
    atoms = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[5, 5, 5], pbc=True)
    low_bulk = CalculatorResult(
        atoms=atoms,
        energy=-1.0,
        energy_above_hull=0.05,
        bulk_modulus=0.0,
    )
    high_bulk = CalculatorResult(
        atoms=atoms,
        energy=-1.0,
        energy_above_hull=0.05,
        bulk_modulus=500.0,
    )

    assert discovery_reward(low_bulk, seen_formulas=set()) == discovery_reward(
        high_bulk, seen_formulas=set()
    )


def test_env_accepts_mlff_model_selector():
    env = gym.make(
        "CrystalDiscovery-v0",
        mlff_model="sevennet",
        max_steps=1,
        max_dataset_rows=1,
    )

    assert isinstance(env.unwrapped.calculator, PhysicalCalculator)
    assert env.unwrapped.mlff_model == "sevennet"


def test_env_rejects_calculator_and_mlff_model_together():
    with pytest.raises(ValueError, match="either calculator"):
        gym.make(
            "CrystalDiscovery-v0",
            calculator=PhysicalCalculator(calculator_factory=LennardJones, relax=False),
            mlff_model="sevennet",
            max_dataset_rows=1,
        )


def test_physical_calculator_reports_real_energy_without_fake_hull():
    atoms = Atoms(
        "Ar2",
        positions=[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]],
        cell=[12.0, 12.0, 12.0],
        pbc=True,
    )
    calculator = PhysicalCalculator(
        calculator_factory=LennardJones,
        element_reference_energies={"Ar": 0.0},
        relax=False,
    )

    result = calculator.evaluate(atoms)
    expected = atoms.copy()
    expected.calc = LennardJones()

    assert result.backend == "physical"
    assert result.energy == expected.get_potential_energy()
    assert result.formation_energy_per_atom == result.energy / 2
    assert result.energy_above_hull is None
    assert result.phonon_stable is None


def test_alex_mp20_dataset_defaults_to_all_splits():
    dataset = AlexMP20Dataset(max_rows=3)

    assert len(dataset) == 3
    if "dataset_split" in dataset.frame:
        assert set(dataset.frame["dataset_split"]).issubset({"train", "val", "test"})


def test_reference_database_builds_same_forcefield_hull_entries(tmp_path):
    records = [
        StructureRecord(
            atoms=Atoms("Ar", positions=[[0.0, 0.0, 0.0]], cell=[10, 10, 10], pbc=True),
            material_id="ar",
            metadata={},
        ),
        StructureRecord(
            atoms=Atoms("Ne", positions=[[0.0, 0.0, 0.0]], cell=[10, 10, 10], pbc=True),
            material_id="ne",
            metadata={},
        ),
        StructureRecord(
            atoms=Atoms("ArNe", positions=[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]], cell=[12, 12, 12], pbc=True),
            material_id="arne",
            metadata={},
        ),
    ]

    database = build_reference_database(
        records,
        calculator_factory=LennardJones,
        cache_path=tmp_path / "refs.json",
    )

    assert set(database.element_reference_energies) == {"Ar", "Ne"}
    assert any(entry.composition == "ArNe" for entry in database.hull_reference_entries)
    assert (tmp_path / "refs.json").exists()


def test_packaged_mlff_checkpoints_are_available():
    assert SUPPORTED_MLFF_MODELS == ("mattersim", "orb", "sevennet")
    for model in SUPPORTED_MLFF_MODELS:
        assert checkpoint_path(model).exists()


def test_build_mlff_calculator_uses_physical_interface():
    calculator = build_mlff_calculator(model="sevennet", relax=False)

    assert isinstance(calculator, PhysicalCalculator)
    assert calculator.relax is False
