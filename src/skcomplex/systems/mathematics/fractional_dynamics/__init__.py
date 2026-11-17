"""Fractional dynamics systems.

The package covers relaxation, oscillators, and chaotic flows whose states have Caputo derivatives of an order.
"""

from ._fractional import (
    FractionalLorenzInteraction,
    FractionalLorenzSystem,
    FractionalOscillatorInteraction,
    FractionalOscillatorSystem,
    FractionalRelaxationInteraction,
    FractionalRelaxationSystem,
    FractionalVanDerPolInteraction,
    FractionalVanDerPolSystem,
)

__all__: list[str] = [
    'FractionalLorenzInteraction',
    'FractionalLorenzSystem',
    'FractionalOscillatorInteraction',
    'FractionalOscillatorSystem',
    'FractionalRelaxationInteraction',
    'FractionalRelaxationSystem',
    'FractionalVanDerPolInteraction',
    'FractionalVanDerPolSystem',
]
