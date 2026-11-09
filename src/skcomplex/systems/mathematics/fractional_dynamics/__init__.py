"""Fractional dynamics systems.

The package covers relaxation, oscillators, and chaotic flows whose states have Caputo derivatives of an order.
"""

from ._fractional import (
    FractionalLorenzSystem,
    FractionalOscillatorSystem,
    FractionalRelaxationSystem,
    FractionalVanDerPolSystem,
    VanDerPolElement,
    van_der_pol_type,
)

__all__: list[str] = [
    'FractionalLorenzSystem',
    'FractionalOscillatorSystem',
    'FractionalRelaxationSystem',
    'FractionalVanDerPolSystem',
    'VanDerPolElement',
    'van_der_pol_type',
]
