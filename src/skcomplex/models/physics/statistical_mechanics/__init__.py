"""Statistical mechanics: gases, fluids, and spin models."""

from ._gases import ideal_gas, lennard_jones_fluid
from ._spins import IsingSpin, PottsSpin, XYSpin, ising, potts, xy

__all__: list[str] = [
    'IsingSpin',
    'PottsSpin',
    'XYSpin',
    'ideal_gas',
    'ising',
    'lennard_jones_fluid',
    'potts',
    'xy',
]
