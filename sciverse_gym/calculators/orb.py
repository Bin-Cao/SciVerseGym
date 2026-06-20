"""ORB ASE calculator adapter.

By Dr.Bin Cao (https://bin-cao.github.io/)
"""

from .checkpoints import checkpoint_path


class ORBCalculatorFactory:
    """Factory for ORB ASE calculators backed by a local checkpoint."""

    def __init__(
        self,
        checkpoint=None,
        device: str = "cpu",
        precision: str = "float32-high",
        compile=None,
        **kwargs,
    ):
        self.checkpoint = checkpoint or checkpoint_path("orb")
        self.device = device
        self.precision = precision
        self.compile = compile
        self.kwargs = dict(kwargs)
        self._model = None

    def __call__(self):
        try:
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator
        except ImportError as exc:
            raise ImportError(
                "Install sciverse-gym with the 'mlff' or 'orb' extra before "
                "using ORBCalculatorFactory."
            ) from exc

        if self._model is None:
            self._model = pretrained.orb_v3_conservative_inf_omat(
                weights_path=str(self.checkpoint),
                device=self.device,
                precision=self.precision,
                compile=self.compile,
            )
        return ORBCalculator(self._model, device=self.device, **self.kwargs)


def orb_calculator_factory(**kwargs):
    """Return a reusable factory for the packaged ORB checkpoint."""
    return ORBCalculatorFactory(**kwargs)
