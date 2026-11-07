"""Fractals: escape-time sets, chaos-game attractors, and words and patches of named curves and sets."""

from ._chaos_game import ChaosPointElement, RandomAffineMapInteraction, chaos_game
from ._quadratic import InverseJuliaPointElement, QuadraticMapElement, inverse_julia, julia, mandelbrot
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
    'ChaosPointElement',
    'InverseJuliaPointElement',
    'KochElement',
    'QuadraticMapElement',
    'RandomAffineMapInteraction',
    'SierpinskiElement',
    'cantor_set',
    'chaos_game',
    'inverse_julia',
    'julia',
    'koch_curve',
    'mandelbrot',
    'sierpinski_substitution',
]
