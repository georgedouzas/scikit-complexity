"""Neuroscience: spiking neurons whose spikes are events."""

from ._neurons import Izhikevich, LeakyIntegrateAndFire, integrate_and_fire, izhikevich

__all__: list[str] = ['Izhikevich', 'LeakyIntegrateAndFire', 'integrate_and_fire', 'izhikevich']
