"""Neuroscience: spiking neurons."""

from ._neurons import Izhikevich, LeakyIntegrateAndFire, integrate_and_fire, izhikevich

__all__: list[str] = [
    'Izhikevich',
    'LeakyIntegrateAndFire',
    'integrate_and_fire',
    'izhikevich',
]
