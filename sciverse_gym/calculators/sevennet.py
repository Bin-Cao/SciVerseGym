"""SevenNet ASE calculator adapter.

By Dr.Bin Cao (https://bin-cao.github.io/)
"""

from .checkpoints import checkpoint_path


class SevenNetCalculatorFactory:
    """Factory for SevenNet ASE calculators backed by a local checkpoint."""

    def __init__(
        self,
        checkpoint=None,
        device: str = "cpu",
        file_type: str = "checkpoint",
        **kwargs,
    ):
        self.checkpoint = checkpoint or checkpoint_path("sevennet")
        self.device = device
        self.file_type = file_type
        self.kwargs = dict(kwargs)

    def __call__(self):
        try:
            from sevenn.calculator import SevenNetCalculator
        except ImportError as exc:
            raise ImportError(
                "Install sciverse-gym with the 'mlff' or 'sevennet' extra "
                "before using SevenNetCalculatorFactory."
            ) from exc

        return SevenNetCalculator(
            model=str(self.checkpoint),
            file_type=self.file_type,
            device=self.device,
            **self.kwargs,
        )


def sevennet_calculator_factory(**kwargs):
    """Return a reusable factory for the packaged SevenNet checkpoint."""
    return SevenNetCalculatorFactory(**kwargs)
