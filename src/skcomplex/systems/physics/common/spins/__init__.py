"""Spin realizations shared by the physics packages: Ising spins and the Monte Carlo sampling of Ising spins."""

from ._ising_sampling import IsingSamplingInteraction
from ._ising_spin import IsingSpinElement
from ._spin_sampling import SpinSamplingInteraction
from ._thermal_sampling import ThermalSamplingInteraction

__all__: list[str] = [
    'IsingSamplingInteraction',
    'IsingSpinElement',
    'SpinSamplingInteraction',
    'ThermalSamplingInteraction',
]
