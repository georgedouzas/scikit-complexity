"""Finance systems: goods moved by randomly walking traders who follow arbitrage opportunities."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement
from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import advection, laplacian
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system


class ArbitrageDensityElement(Element):
    """Density of a good carried by random-walk traders, minimally coupled to an exchange rate connection.

    Args:
        label:
            The label of the density.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        rho:
            The density of the good.

        D:
            The diffusion constant of the traders.

        A_x:
            The initial spatial component of the exchange rate connection.

        E:
            The arbitrage field strength, the growth rate of the spatial component.

        A_t:
            The temporal component of the exchange rate connection.

    Examples:
        >>> from skcomplex.systems.economics.finance import ArbitrageDensityElement
        >>> ArbitrageDensityElement('market', E=1.0).E
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        D: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        A_x: Annotated[object, ParameterSpec()] = 0.0,
        E: Annotated[object, ParameterSpec()] = 0.5,
        A_t: Annotated[object, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho
        self.D = D
        self.A_x = A_x
        self.E = E
        self.A_t = A_t


def _covariant_diffusion(s: SimpleNamespace) -> Expression:
    """Return `D (d_x - A_x(t))^2 rho + A_t rho`, with the connection `A_x(t) = A_x + E t`."""
    market = s.target
    connection = market.A_x + market.E * s.t
    return (
        market.D * laplacian(market.rho)
        - advection(market.rho, [2 * market.D * connection])
        + (market.D * connection**2 + market.A_t) * market.rho
    )


@register_system(
    field='economics/finance',
    tags=('arbitrage', 'diffusion', 'gauge field', 'pde'),
    references=('Schwichtenberg 2020',),
    defaults={'until': 2.0, 'dt': 0.1},
)
def arbitrage_traders(
    n_sites: int = 200,
    *,
    length: float = 20.0,
    D: float = 0.5,
    A_x: float = 0.0,
    E: float = 0.5,
    A_t: float = 0.0,
) -> System:
    """Return Schwichtenberg's diffusion of a good by random-walk traders, minimally coupled to exchange rates.

    The density of the good follows `(d_t - A_t) rho = D (d_x - A_x(t))^2 rho` on a periodic line, with the exchange
    rate connection `A_x(t) = A_x + E t`. The arbitrage opportunity is the field strength `F_tx = E`. Without it, the
    traders diffuse the good symmetrically, and with it, they drift at the velocity `2 D A_x(t)`.

    Args:
        n_sites:
            The number of sites of the line.

        length:
            The length of the line.

        D:
            The diffusion constant of the traders.

        A_x:
            The initial spatial component of the exchange rate connection.

        E:
            The arbitrage field strength, the growth rate of the spatial component.

        A_t:
            The temporal component of the exchange rate connection.

    Returns:
        system:
            The system, with the density of the good `market.rho`.

    Examples:
        >>> from skcomplex.systems.economics.finance import arbitrage_traders
        >>> arbitrage_traders(50).elements[0].values['E']
        0.5
    """
    spacing = length / n_sites
    positions = (np.arange(n_sites) + 0.5) * spacing - length / 2
    rho = np.exp(-(positions**2)) / np.sqrt(np.pi)
    values = {'rho': rho, 'D': D, 'A_x': A_x, 'E': E, 'A_t': A_t}
    line = LatticeSpace(n_sites, spacing=spacing, boundary='periodic')
    market = FieldElement('market', element=ArbitrageDensityElement, space=line, values=values)
    transport = RateInteraction('transport', target='market', state='rho', rate=_covariant_diffusion)
    return System(elements=[market], interactions=[transport])
