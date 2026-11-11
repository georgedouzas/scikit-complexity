"""Finance systems: goods moved by randomly walking traders.

The traders follow arbitrage opportunities.
"""

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
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ArbitrageDensityElement(Element):
    """Density of a good carried by random-walk traders.

    It is minimally coupled to an exchange rate connection.

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


class ArbitrageTradersSystem(BaseSystem):
    """Schwichtenberg's diffusion of a good by random-walk traders.

    It is minimally coupled to exchange rates. The density of the good follows
    `(d_t - A_t) rho = D (d_x - A_x(t))^2 rho` on a periodic line, with the exchange rate connection
    `A_x(t) = A_x + E t`. The arbitrage opportunity is the field strength `F_tx = E`. Without it, the traders diffuse
    the good symmetrically, and with it, they drift at the velocity `2 D A_x(t)`. The system has the density of the good
    `market.rho`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.finance import ArbitrageTradersSystem
        >>> ArbitrageTradersSystem(50).prepare().system_.elements[0].values['E']
        0.5
    """

    field = 'economics/finance'
    topics = ('arbitrage', 'diffusion', 'gauge field', 'pde')
    references = ('Schwichtenberg 2020',)
    until = 2.0
    dt = 0.1

    def __init__(
        self,
        n_sites: int = 200,
        *,
        length: float = 20.0,
        D: float = 0.5,
        A_x: float = 0.0,
        E: float = 0.5,
        A_t: float = 0.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_sites = n_sites
        self.length = length
        self.D = D
        self.A_x = A_x
        self.E = E
        self.A_t = A_t

    @staticmethod
    def _get_covariant_diffusion(s: SimpleNamespace) -> Expression:
        """Return `D (d_x - A_x(t))^2 rho + A_t rho`.

        The connection is `A_x(t) = A_x + E t`.
        """
        market = s.target
        connection = market.A_x + market.E * s.t
        return (
            market.D * laplacian(market.rho)
            - advection(market.rho, [2 * market.D * connection])
            + (market.D * connection**2 + market.A_t) * market.rho
        )

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        spacing = self.length / self.n_sites
        positions = (np.arange(self.n_sites) + 0.5) * spacing - self.length / 2
        rho = np.exp(-(positions**2)) / np.sqrt(np.pi)
        values = {'rho': rho, 'D': self.D, 'A_x': self.A_x, 'E': self.E, 'A_t': self.A_t}
        line = LatticeSpace(self.n_sites, spacing=spacing, boundary='periodic')
        market = FieldElement('market', element=ArbitrageDensityElement, space=line, values=values)
        transport = RateInteraction('transport', target='market', state='rho', rate=self._get_covariant_diffusion)
        return CompositeSystem(self.label, elements=[market], interactions=[transport])
