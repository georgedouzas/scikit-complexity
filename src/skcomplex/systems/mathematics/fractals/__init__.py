"""Fractal systems.

The package covers escape-time sets, chaos-game attractors, and words and patches of named curves and sets.
"""

from ._chaos_game import (
    ChaosGameSystem,
    ChaosPointElement,
    RandomAffineMapInteraction,
)
from ._quadratic import (
    InverseJuliaInteraction,
    InverseJuliaPointElement,
    InverseJuliaSystem,
    JuliaSystem,
    MandelbrotSystem,
    QuadraticMapElement,
    QuadraticMapInteraction,
)
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
    'InverseJuliaInteraction',
    'InverseJuliaPointElement',
    'InverseJuliaSystem',
    'JuliaSystem',
    'KochCurveSystem',
    'KochElement',
    'MandelbrotSystem',
    'QuadraticMapElement',
    'QuadraticMapInteraction',
    'RandomAffineMapInteraction',
    'SierpinskiElement',
    'SierpinskiSubstitutionSystem',
]
