"""Fractals: escape-time sets, chaos-game attractors, and words and patches of named curves and sets."""

from ._chaos_game import ChaosGameSystem, ChaosPointElement, RandomAffineMapInteraction
from ._quadratic import InverseJuliaPointElement, InverseJuliaSystem, JuliaSystem, MandelbrotSystem, QuadraticMapElement
from ._substitution import (
    CantorElement,
    KochElement,
    SierpinskiElement,
    cantor_set,
    koch_curve,
    sierpinski_substitution,
)

__all__: list[str] = [
    'CantorElement',
    'ChaosGameSystem',
    'ChaosPointElement',
    'InverseJuliaPointElement',
    'InverseJuliaSystem',
    'JuliaSystem',
    'KochElement',
    'MandelbrotSystem',
    'QuadraticMapElement',
    'RandomAffineMapInteraction',
    'SierpinskiElement',
    'cantor_set',
    'koch_curve',
    'sierpinski_substitution',
]
