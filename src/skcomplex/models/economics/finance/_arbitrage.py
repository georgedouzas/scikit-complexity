"""Finance models: goods moved by randomly walking traders who follow arbitrage opportunities."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, FieldElement
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import advection, laplacian
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model


class ArbitrageDensityElement(Element):
    """Arbitrage density element."""

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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the arbitrage density element."""
        return [
            RateLaw(
                s.rho,
                s.D * laplacian(s.rho)
                - advection(s.rho, [2 * s.D * (s.A_x + s.E * s.t)])
                + (s.D * (s.A_x + s.E * s.t) ** 2 + s.A_t) * s.rho,
            ),
        ]


@register_model(
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
            The model, with the density of the good `market.rho`.

    Examples:
        >>> from skcomplex.models.economics.finance import arbitrage_traders
        >>> arbitrage_traders(50).elements[0].values['E']
        0.5
    """
    spacing = length / n_sites
    positions = (np.arange(n_sites) + 0.5) * spacing - length / 2
    rho = np.exp(-(positions**2)) / np.sqrt(np.pi)
    values = {'rho': rho, 'D': D, 'A_x': A_x, 'E': E, 'A_t': A_t}
    line = LatticeSpace(n_sites, spacing=spacing, boundary='periodic')
    market = FieldElement('market', element=ArbitrageDensityElement, space=line, values=values)
    return System(elements=[market])
