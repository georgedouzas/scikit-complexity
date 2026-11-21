"""Finance systems: goods moved by randomly walking traders.

The traders follow arbitrage opportunities.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import advection, laplacian
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class ArbitrageDensityElement(Element):
    """Density of a good carried by random-walk traders.

    It is minimally coupled to an exchange rate connection.

    Args:
        label:
            The label of the density.

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

        space:
            The line whose sites hold the density, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.economics.finance import ArbitrageDensityElement
        >>> ArbitrageDensityElement('market', E=1.0).E
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        rho: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        D: Annotated[object, RealSet.open(0, oo)] = 0.5,
        A_x: object = 0.0,
        E: object = 0.5,
        A_t: object = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.rho = rho
        self.D = D
        self.A_x = A_x
        self.E = E
        self.A_t = A_t


class CovariantDiffusionInteraction(Interaction):
    """Diffusion of a density by random-walk traders, minimally coupled to an exchange rate connection.

    The density follows `(d_t - A_t) rho = D (d_x - A_x(t))^2 rho`, with the connection `A_x(t) = A_x + E t`, so it
    diffuses at the rate `D`, drifts at the velocity `2 D A_x(t)`, and grows at the rate `D A_x(t)^2 + A_t`.

    Args:
        label:
            The label of the interaction.

        target:
            The density of the good, with the diffusion constant and the connection among its values.

    Examples:
        >>> from skcomplex.systems.economics.finance import CovariantDiffusionInteraction
        >>> CovariantDiffusionInteraction('transport').target
        'market'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role = 'market') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the covariant diffusion of the density as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            market = context.target
            connection = market.A_x + market.E * context.t
            rate = (
                market.D * laplacian(market.rho)
                - advection(market.rho, [2 * market.D * connection])
                + (market.D * connection**2 + market.A_t) * market.rho
            )
            self.laws_: list[BaseLaw] = [RateLaw(market.rho, rate)]


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.finance import ArbitrageTradersSystem
        >>> ArbitrageTradersSystem(50).prepare().system_.elements[0].E
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_sites = n_sites
        self.length = length
        self.D = D
        self.A_x = A_x
        self.E = E
        self.A_t = A_t

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a Gaussian density of the good on a periodic line and its transport."""
        super()._prepare(context)
        spacing = self.length / self.n_sites
        positions = (np.arange(self.n_sites) + 0.5) * spacing - self.length / 2
        rho = np.exp(-(positions**2)) / np.sqrt(np.pi)
        line = LatticeSpace(self.n_sites, spacing=spacing, boundary='periodic')
        market = ArbitrageDensityElement('market', rho=rho, D=self.D, A_x=self.A_x, E=self.E, A_t=self.A_t, space=line)
        transport = CovariantDiffusionInteraction('transport', target='market')
        self.system_ = System(self.label, elements=[market], interactions=[transport])
