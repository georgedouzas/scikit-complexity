"""Excitable kinetics shared by the biology packages: FitzHugh-Nagumo kinetics, theta neurons, and spike resets."""

from ._fitzhugh_nagumo import FitzHughNagumoInteraction
from ._spike_reset import SpikeResetInteraction
from ._theta_neuron import ThetaNeuronElement

__all__: list[str] = [
    'FitzHughNagumoInteraction',
    'SpikeResetInteraction',
    'ThetaNeuronElement',
]
