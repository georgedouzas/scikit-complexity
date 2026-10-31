"""Spin realizations shared by the physics packages: Monte Carlo sampling of Ising, Potts, and XY spins."""

from ._ising_sampling import IsingSamplingInteraction
from ._potts_sampling import PottsSamplingInteraction
from ._spin_sampling import SpinSamplingInteraction
from ._xy_sampling import XYSamplingInteraction

__all__: list[str] = [
    'IsingSamplingInteraction',
    'PottsSamplingInteraction',
    'SpinSamplingInteraction',
    'XYSamplingInteraction',
]
