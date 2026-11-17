"""Oscillator realizations shared by the physics packages.

The package holds phase oscillators, links, and order.
"""

from ._kuramoto_coupling import KuramotoCouplingInteraction
from ._link import LinkElement
from ._order_parameter import order_parameter
from ._phase_oscillator import PhaseOscillatorElement
from ._phase_rotation import PhaseRotationInteraction
from ._sited_oscillator import SitedOscillatorElement

__all__: list[str] = [
    'KuramotoCouplingInteraction',
    'LinkElement',
    'PhaseOscillatorElement',
    'PhaseRotationInteraction',
    'SitedOscillatorElement',
    'order_parameter',
]
