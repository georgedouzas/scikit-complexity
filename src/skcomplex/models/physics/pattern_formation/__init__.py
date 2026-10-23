"""Pattern formation: amplitude equations, rolls, phase ordering, and nonreciprocal conserved fields."""

from ._conserved import NonreciprocalMixture, nonreciprocal_cahn_hilliard
from ._ordering import AllenCahn, CahnHilliard, allen_cahn, cahn_hilliard
from ._patterns import GinzburgLandau, SwiftHohenberg, complex_ginzburg_landau, swift_hohenberg

__all__: list[str] = [
    'AllenCahn',
    'CahnHilliard',
    'GinzburgLandau',
    'NonreciprocalMixture',
    'SwiftHohenberg',
    'allen_cahn',
    'cahn_hilliard',
    'complex_ginzburg_landau',
    'nonreciprocal_cahn_hilliard',
    'swift_hohenberg',
]
