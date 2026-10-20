"""Fractals: escape-time sets of the complex quadratic map, attractors drawn by the chaos game, and substitutions."""

from ._chaos_game import ChaosPoint, chaos_game
from ._quadratic import QuadraticMap, julia, mandelbrot
from ._substitution import cantor_set, koch_curve, sierpinski_substitution

__all__: list[str] = [
    'ChaosPoint',
    'QuadraticMap',
    'cantor_set',
    'chaos_game',
    'julia',
    'koch_curve',
    'mandelbrot',
    'sierpinski_substitution',
]
