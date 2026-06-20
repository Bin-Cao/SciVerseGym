"""MatterSim ASE calculator adapter.

By Dr.Bin Cao (https://bin-cao.github.io/)
"""

from .checkpoints import checkpoint_path


class MatterSimCalculatorFactory:
    """Factory for MatterSim ASE calculators backed by a local checkpoint."""

    def __init__(
        self,
        checkpoint=None,
        device: str = "cpu",
        compute_stress: bool = True,
        **kwargs,
    ):
        self.checkpoint = checkpoint or checkpoint_path("mattersim")
        self.device = device
        self.compute_stress = compute_stress
        self.kwargs = dict(kwargs)

    def __call__(self):
        try:
            from mattersim.forcefield import MatterSimCalculator
        except ImportError as exc:
            raise ImportError(
                "Install sciverse-gym with the 'mlff' or 'mattersim' extra "
                "before using MatterSimCalculatorFactory."
            ) from exc

        return MatterSimCalculator.from_checkpoint(
            str(self.checkpoint),
            device=self.device,
            compute_stress=self.compute_stress,
            **self.kwargs,
        )


def mattersim_calculator_factory(**kwargs):
    """Return a reusable factory for the packaged MatterSim checkpoint."""
    return MatterSimCalculatorFactory(**kwargs)
