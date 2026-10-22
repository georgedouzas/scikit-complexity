"""Thermohaline circulation between ocean boxes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import abs_symbolic

from ....core.components import Element
from ....core.laws import Rate
from ....core.runtime import register_model
from ....core.system import System

StommelBoxes = Element.define(
    'StommelBoxes',
    states={'T': 3.0, 'S': 0.0},
    parameters={'eta1': 3.0, 'eta2': 1.2, 'eta3': 0.3},
    laws=lambda s: [
        Rate(s.T, s.eta1 - s.T * (1 + abs_symbolic(s.T - s.S))),
        Rate(s.S, s.eta2 - s.S * (s.eta3 + abs_symbolic(s.T - s.S))),
    ],
)


@register_model(
    field='earth/oceanography',
    tags=('ode', 'bistability', 'thermohaline circulation', 'tipping'),
    references=('Stommel 1961', 'Cessi 1994', 'Ghil and Lucarini 2020'),
    defaults={'until': 50.0, 'dt': 0.1},
)
def stommel(
    T: float = 3.0,
    S: float = 0.0,
    *,
    eta1: float = 3.0,
    eta2: float = 1.2,
    eta3: float = 0.3,
) -> System:
    """Return Stommel's two-box thermohaline circulation, bistable between a thermal and a haline mode.

    The dimensionless differences of temperature `T` and salinity `S` between an equatorial and a polar box relax to
    their forcing and are mixed by a flow proportional to the density difference `|T - S|`:
    `dT/dt = eta1 - T (1 + |T - S|)` and `dS/dt = eta2 - S (eta3 + |T - S|)`. For the default parameters a strong
    circulation driven by temperature and a weak one driven by salinity are both stable.

    Args:
        T:
            The initial temperature difference.

        S:
            The initial salinity difference.

        eta1:
            The thermal forcing.

        eta2:
            The freshwater forcing.

        eta3:
            The ratio of the relaxation times of temperature and salinity.

    Returns:
        system:
            The model, with the differences `ocean.T` and `ocean.S`.

    Examples:
        >>> from skcomplex.models.earth.oceanography import stommel
        >>> stommel(eta2=1.0).elements[0].eta2
        1.0
    """
    return System(components=[StommelBoxes('ocean', T=T, S=S, eta1=eta1, eta2=eta2, eta3=eta3)])
