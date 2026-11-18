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
    CantorInteraction,
    CantorSetSystem,
    KochCurveSystem,
    SierpinskiInteraction,
    SierpinskiSubstitutionSystem,
)

__all__: list[str] = [
    'CantorInteraction',
    'CantorSetSystem',
    'ChaosGameSystem',
    'ChaosPointElement',
    'InverseJuliaInteraction',
    'InverseJuliaPointElement',
    'InverseJuliaSystem',
    'JuliaSystem',
    'KochCurveSystem',
    'MandelbrotSystem',
    'QuadraticMapElement',
    'QuadraticMapInteraction',
    'RandomAffineMapInteraction',
    'SierpinskiInteraction',
    'SierpinskiSubstitutionSystem',
]
