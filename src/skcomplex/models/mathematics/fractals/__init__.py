"""Fractals: escape-time sets of the complex quadratic map, attractors drawn by the chaos game, and substitutions."""

from ._chaos_game import ChaosPointElement, chaos_game
from ._quadratic import QuadraticMapElement, julia, mandelbrot
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
    'KochElement',
    'QuadraticMapElement',
    'SierpinskiElement',
    'cantor_set',
    'chaos_game',
    'julia',
    'koch_curve',
    'mandelbrot',
    'sierpinski_substitution',
]
