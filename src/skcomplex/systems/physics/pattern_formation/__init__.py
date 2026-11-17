"""Pattern formation systems.

The package covers amplitude equations, rolls, phase ordering, and nonreciprocal conserved fields.
"""

from ._conserved import NonreciprocalCahnHilliardSystem, NonreciprocalMixtureElement
from ._ordering import AllenCahnSystem, CahnHilliardSystem
from ._patterns import (
    ComplexGinzburgLandauSystem,
    GinzburgLandauElement,
    GinzburgLandauInteraction,
    PatternFieldElement,
    SwiftHohenbergSystem,
)

__all__: list[str] = [
    'AllenCahnSystem',
    'CahnHilliardSystem',
    'ComplexGinzburgLandauSystem',
    'GinzburgLandauElement',
    'GinzburgLandauInteraction',
    'NonreciprocalCahnHilliardSystem',
    'NonreciprocalMixtureElement',
    'PatternFieldElement',
    'SwiftHohenbergSystem',
]
