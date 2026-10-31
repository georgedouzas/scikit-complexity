"""Oscillator realizations shared by the physics packages: phase, Stuart-Landau, and Van der Pol oscillators."""

from ._kuramoto_coupling import KuramotoCouplingInteraction
from ._phase_oscillator import PhaseOscillatorElement
from ._sited_oscillator import SitedOscillatorElement
from ._stuart_landau import StuartLandauElement
from ._van_der_pol import VanDerPolElement

__all__: list[str] = [
    'KuramotoCouplingInteraction',
    'PhaseOscillatorElement',
    'SitedOscillatorElement',
    'StuartLandauElement',
    'VanDerPolElement',
]
