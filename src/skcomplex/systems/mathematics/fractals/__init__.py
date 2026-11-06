"""Fractals: escape-time sets, chaos-game attractors, and words and patches of named curves and sets."""

from ._cantor_element import CantorElement
from ._chaos_game import ChaosPointElement, chaos_game
from ._inverse_julia_point import InverseJuliaPointElement
from ._koch_element import KochElement
from ._quadratic import QuadraticMapElement, inverse_julia, julia, mandelbrot
from ._random_affine_map import RandomAffineMapInteraction
from ._sierpinski_element import SierpinskiElement
from ._substitution import cantor_set, koch_curve, sierpinski_substitution

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
