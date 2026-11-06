"""Pattern formation: amplitude equations, rolls, phase ordering, and nonreciprocal conserved fields."""

from ._conserved import NonreciprocalMixtureElement, nonreciprocal_cahn_hilliard
from ._ginzburg_landau import GinzburgLandauElement
from ._ordering import allen_cahn, cahn_hilliard
from ._pattern_field import PatternFieldElement
from ._patterns import complex_ginzburg_landau, swift_hohenberg

__all__: list[str] = [
    'GinzburgLandauElement',
    'NonreciprocalMixtureElement',
    'PatternFieldElement',
    'allen_cahn',
    'cahn_hilliard',
    'complex_ginzburg_landau',
    'nonreciprocal_cahn_hilliard',
    'swift_hohenberg',
]
