"""Pattern formation: amplitude equations, rolls, phase ordering, and nonreciprocal conserved fields."""

from ._conserved import NonreciprocalMixtureComponent, nonreciprocal_cahn_hilliard
from ._ordering import AllenCahnComponent, CahnHilliardComponent, allen_cahn, cahn_hilliard
from ._patterns import GinzburgLandauComponent, SwiftHohenbergComponent, complex_ginzburg_landau, swift_hohenberg

__all__: list[str] = [
    'AllenCahnComponent',
    'CahnHilliardComponent',
    'GinzburgLandauComponent',
    'NonreciprocalMixtureComponent',
    'SwiftHohenbergComponent',
    'allen_cahn',
    'cahn_hilliard',
    'complex_ginzburg_landau',
    'nonreciprocal_cahn_hilliard',
    'swift_hohenberg',
]
