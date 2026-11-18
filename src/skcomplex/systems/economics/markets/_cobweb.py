"""Markets systems: price dynamics of supply and demand."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import Role
from ....core.systems import BaseSystem, System


class MarketElement(Element):
    """Market of one good described by its price.

    Args:
        label:
            The label of the market.

        price:
            The initial price.

    Examples:
        >>> from skcomplex.systems.economics.markets import MarketElement
        >>> MarketElement('wheat', price=2.0).price
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        price: float = 1.0,
    ) -> None:
        super().__init__(label)
        self.price = price


class CobwebInteraction(Interaction):
    """Cobweb system, where supply reacts to the previous price.

    Demand clears the market. Demand is `a - b * price` and supply is `c + d * previous price`, so the next price is
    `(a - c - d * price) / b`, a map of the state `price` of the market.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the market.

        a:
            The demand intercept.

        b:
            The demand slope.

        c:
            The supply intercept.

        d:
            The supply slope.

    Examples:
        >>> from skcomplex.systems.economics.markets import CobwebInteraction
        >>> CobwebInteraction('cobweb', target='wheat', a=10, b=2, c=1, d=1).b
        2
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'market',
        a: float = 10.0,
        b: Annotated[float, RealSet.open(0, oo)] = 2.0,
        c: float = 1.0,
        d: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the next price `(a - c - d * price) / b`, at which demand buys the supply of the last price."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            self.laws_: list[BaseComponent] = [UpdateLaw(s.target.price, (s.a - s.c - s.d * s.target.price) / s.b)]


class CobwebSystem(BaseSystem):
    """A market whose price converges to or oscillates around equilibrium.

    Supply lags the price. The system has the market `market`.

    Args:
        price:
            The initial price.

        a:
            The demand intercept.

        b:
            The demand slope.

        c:
            The supply intercept.

        d:
            The supply slope.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.markets import CobwebSystem
        >>> CobwebSystem(price=2.0).prepare().simulate(until=2).get_values('market.price').tolist()
        [2.0, 3.5, 2.75]
    """

    field = 'economics/markets'
    topics = ('market', 'discrete map')
    references = ('Ezekiel 1938',)
    until = 30
    dt = 1

    def __init__(
        self,
        price: float = 1.0,
        *,
        a: float = 10.0,
        b: float = 2.0,
        c: float = 1.0,
        d: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.price = price
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the market and its cobweb."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[MarketElement('market', price=self.price)],
            interactions=[CobwebInteraction('cobweb', target='market', a=self.a, b=self.b, c=self.c, d=self.d)],
        )
