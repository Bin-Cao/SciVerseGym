"""CrystalDiscovery-v0 environment.

By Dr.Bin Cao (https://bin-cao.github.io/)

This file implements the first-generation SciVerseGym environment for crystal
structure search through structured materials edits. Recommended usage passes
an MLFF backend explicitly in `gym.make`, for example
`mlff_model="sevennet"`.
"""

from typing import Dict, Optional

import numpy as np
from gymnasium import Env, spaces

from sciverse_gym.calculators import SurrogateCalculator, build_mlff_calculator
from sciverse_gym.datasets import AlexMP20Dataset
from sciverse_gym.envs.actions import apply_action
from sciverse_gym.observations import atoms_to_graph
from sciverse_gym.rewards import discovery_reward
from sciverse_gym.structures import canonical_formula


class AnySpace(spaces.Space):
    """Gymnasium space for rich Python objects in scientific observations."""

    def sample(self):
        """Return a placeholder sample for non-array observation fields."""
        return None

    def contains(self, x):
        """Accept any rich Python object stored in the observation."""
        return True


class CrystalDiscoveryEnv(Env):
    """Gymnasium-style crystal discovery environment.

    Use `gym.make("CrystalDiscovery-v0", mlff_model="sevennet", ...)` for the
    default documented workflow. A custom `calculator=` may be supplied instead
    of `mlff_model`; the two options are mutually exclusive.

    `gym.make("CrystalDiscovery-v0", ...)` accepts:

    - `data_path`: ALEX-MP-20 parquet directory. Default:
      `"data/alex-mp-20"`.
    - `split`: dataset split. Default: `"all"`.
    - `max_steps`: episode edit budget. Default: `100`.
    - `calculator`: custom evaluator such as a `PhysicalCalculator`; mutually
      exclusive with `mlff_model`.
    - `mlff_model`: packaged MLFF backend, one of `"sevennet"`,
      `"mattersim"`, or `"orb"`. Recommended examples use `"sevennet"`.
    - `mlff_relax`: relax atomic positions before evaluation. Default:
      `False`.
    - `mlff_relax_cell`: allow cell relaxation when relaxing. Default:
      `False`.
    - `mlff_compute_phonons`: compute finite-displacement phonon stability.
      Default: `False`.
    - `mlff_kwargs`: extra MLFF/physical options such as `relax_fmax`,
      `relax_steps`, `phonon_supercell`, `phonon_displacement`,
      `phonon_mesh`, and `phonon_stability_tolerance`.
    - `max_dataset_rows`: optional dataset row cap for tutorials/tests.
      Default: `None`.
    - `dataset`: mounted dataset object implementing `__len__`, `get(index)`,
      and `sample(rng)`.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1, "name": "CrystalDiscovery-v0"}

    def __init__(
        self,
        data_path="data/alex-mp-20",
        split="all",
        max_steps: int = 100,
        calculator=None,
        mlff_model: Optional[str] = None,
        mlff_relax: bool = False,
        mlff_relax_cell: bool = False,
        mlff_compute_phonons: bool = False,
        mlff_kwargs: Optional[Dict] = None,
        max_dataset_rows: Optional[int] = None,
        dataset=None,
    ):
        """Create a crystal discovery environment.

        These arguments are the keyword arguments users pass through
        `gym.make("CrystalDiscovery-v0", ...)`. The recommended documented
        path is to pass `mlff_model="sevennet"` or another supported MLFF
        explicitly. `calculator=` is reserved for custom or preconfigured
        physical calculators.

        Args:
            data_path: Directory containing the ALEX-MP-20 parquet data.
                Default is `"data/alex-mp-20"`.
            split: Dataset split name. Default is `"all"`, which uses the
                merged local table.
            max_steps: Maximum number of structured edits per episode. When
                the step count reaches this value, `step` returns
                `truncated=True`. Default is `100`.
            calculator: Optional custom evaluator implementing
                `evaluate(atoms, reference=None) -> CalculatorResult`, such as
                a `PhysicalCalculator` with same-force-field hull references.
                Do not pass this together with `mlff_model`.
            mlff_model: Packaged MLFF backend selected inside `gym.make`.
                Supported values are `"sevennet"`, `"mattersim"`, and `"orb"`.
                Recommended examples use `"sevennet"`. If this is omitted and
                `calculator` is also omitted, the legacy deterministic
                `SurrogateCalculator` compatibility path is used.
            mlff_relax: Whether to relax atomic positions before every reset
                and step evaluation. Passed to `build_mlff_calculator` as
                `relax`. Default is `False`.
            mlff_relax_cell: Whether relaxation may change the lattice cell.
                Only relevant when `mlff_relax=True`. Passed as `relax_cell`.
                Default is `False`.
            mlff_compute_phonons: Whether to compute finite-displacement phonon
                stability after each MLFF evaluation. This can be expensive.
                Passed as `compute_phonons`. Default is `False`.
            mlff_kwargs: Extra keyword arguments passed to the packaged MLFF
                factory and `PhysicalCalculator` builder. Common physical
                options include `relax_fmax`, `relax_steps`,
                `phonon_supercell`, `phonon_displacement`, `phonon_mesh`, and
                `phonon_stability_tolerance`; model adapters may also accept
                backend-specific options.
            max_dataset_rows: Optional cap on loaded dataset rows, useful for
                tutorials, smoke tests, and small Jupyter experiments. Default
                is `None`, which loads all available rows for the selected
                split.
            dataset: Optional mounted dataset object. When supplied, it
                replaces `data_path`/`split` loading. The object should
                implement `__len__`, `get(index)`, and `sample(rng)`, and
                return `StructureRecord` instances.

        Raises:
            ValueError: If both `calculator` and `mlff_model` are supplied.
        """
        if calculator is not None and mlff_model is not None:
            raise ValueError("Pass either calculator=... or mlff_model=..., not both.")
        self.dataset = dataset or AlexMP20Dataset(
            data_path=data_path,
            split=split,
            max_rows=max_dataset_rows,
        )
        self.max_steps = int(max_steps)
        self.mlff_model = mlff_model
        if calculator is not None:
            self.calculator = calculator
        elif mlff_model is not None:
            self.calculator = build_mlff_calculator(
                model=mlff_model,
                relax=mlff_relax,
                relax_cell=mlff_relax_cell,
                compute_phonons=mlff_compute_phonons,
                **dict(mlff_kwargs or {}),
            )
        else:
            self.calculator = SurrogateCalculator()
        self.rng = np.random.default_rng()
        self.current_atoms = None
        self.current_record = None
        self.current_result = None
        self.step_count = 0
        self.seen_formulas = set()

        self.action_space = spaces.Dict(
            {
                "action_type": spaces.Discrete(5),
                "site": spaces.Discrete(512),
                "element": spaces.Discrete(95),
                "lattice_delta": spaces.Box(low=-0.15, high=0.15, shape=(3,), dtype=np.float32),
                "atom_delta": spaces.Box(low=-0.25, high=0.25, shape=(3,), dtype=np.float32),
                "insert_position": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "structure": AnySpace(),
                "graph": AnySpace(),
                "energy": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                "step": spaces.Discrete(max(self.max_steps + 1, 1)),
                "budget_left": spaces.Discrete(max(self.max_steps + 1, 1)),
                "metadata": AnySpace(),
            }
        )

    def reset(self, *, seed=None, options=None):
        """Reset to a sampled crystal and evaluate it with the configured backend.

        The backend is the MLFF selected by `mlff_model` in `gym.make(...)`, or
        the custom `calculator=` supplied by the caller.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        if "index" in options:
            self.current_record = self.dataset.get(int(options["index"]))
        else:
            self.current_record = self.dataset.sample(self.rng)
        self.current_atoms = self.current_record.atoms.copy()
        self.current_result = self.calculator.evaluate(
            self.current_atoms, reference=self.current_record.metadata
        )
        self.current_atoms = self.current_result.atoms.copy()
        self.step_count = 0
        self.seen_formulas = {canonical_formula(self.current_atoms)}
        return self._observation(), self._info(reward=0.0)

    def step(self, action: Dict):
        """Submit one structured action and return the new state transition.

        The action is applied to the current crystal, the configured MLFF or
        custom calculator evaluates the edited structure, and the method returns
        `(obs, reward, terminated, truncated, info)`.
        """
        if self.current_atoms is None:
            self.reset()

        candidate = apply_action(self.current_atoms, action)
        result = self.calculator.evaluate(candidate, reference=self.current_record.metadata)
        reward = discovery_reward(result, self.seen_formulas)

        self.current_result = result
        self.current_atoms = result.atoms.copy()
        self.step_count += 1
        self.seen_formulas.add(canonical_formula(self.current_atoms))

        terminated = False
        truncated = self.step_count >= self.max_steps
        return self._observation(), reward, terminated, truncated, self._info(reward=reward)

    def render(self):
        """Render the current formula, energy, hull value, and step counter."""
        if self.current_atoms is None:
            return "CrystalDiscoveryEnv(uninitialized)"
        hull = self.current_result.energy_above_hull
        hull_text = "not_computed" if hull is None else f"{hull:.4f}"
        return (
            f"{self.current_atoms.get_chemical_formula()} "
            f"energy={self.current_result.energy:.4f} "
            f"hull={hull_text} "
            f"step={self.step_count}/{self.max_steps}"
        )

    def _observation(self):
        """Build the current Gymnasium observation dictionary."""
        return {
            "structure": self.current_atoms.copy(),
            "graph": atoms_to_graph(self.current_atoms),
            "energy": np.asarray(self.current_result.energy, dtype=np.float32),
            "step": int(self.step_count),
            "budget_left": int(max(self.max_steps - self.step_count, 0)),
            "metadata": {
                "material_id": self.current_record.material_id,
                **self.current_record.metadata,
            },
        }

    def _info(self, reward: float):
        """Build transition metadata returned by `reset` and `step`."""
        return {
            "reward": float(reward),
            "backend": self.current_result.backend if self.current_result else None,
            "energy_above_hull": self.current_result.energy_above_hull if self.current_result else None,
            "formation_energy_per_atom": (
                self.current_result.formation_energy_per_atom if self.current_result else None
            ),
            "phonon_stable": self.current_result.phonon_stable if self.current_result else None,
            "min_phonon_frequency": (
                self.current_result.min_phonon_frequency if self.current_result else None
            ),
            "bulk_modulus": self.current_result.bulk_modulus if self.current_result else None,
            "formula": canonical_formula(self.current_atoms) if self.current_atoms is not None else None,
        }
