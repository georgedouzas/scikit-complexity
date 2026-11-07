"""Fractals: escape-time sets, chaos-game attractors, and words and patches of named curves and sets."""

from ._chaos_game import ChaosGameSystem, ChaosPointElement, RandomAffineMapInteraction
from ._quadratic import InverseJuliaPointElement, InverseJuliaSystem, JuliaSystem, MandelbrotSystem, QuadraticMapElement
from ._substitution import (
    CantorElement,
    CantorSetSystem,
    KochCurveSystem,
    KochElement,
    SierpinskiElement,
    SierpinskiSubstitutionSystem,
)

__all__: list[str] = [
    'CantorElement',
    'CantorSetSystem',
    'ChaosGameSystem',
    'ChaosPointElement',
    'InverseJuliaPointElement',
    'InverseJuliaSystem',
    'JuliaSystem',
    'KochCurveSystem',
    'KochElement',
    'MandelbrotSystem',
    'QuadraticMapElement',
    'RandomAffineMapInteraction',
    'SierpinskiElement',
    'SierpinskiSubstitutionSystem',
]
