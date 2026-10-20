"""Fractals: escape-time sets of the complex quadratic map and attractors drawn by the chaos game."""

from ._chaos_game import ChaosPoint, chaos_game
from ._quadratic import QuadraticMap, julia, mandelbrot

__all__: list[str] = [
    'ChaosPoint',
    'QuadraticMap',
    'chaos_game',
    'julia',
    'mandelbrot',
]
