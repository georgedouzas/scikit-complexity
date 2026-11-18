"""Natural resources systems: harvesting renewable stocks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw, UpdateLaw
from ....core.components.laws.utils import count
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import AgentView, BaseSolver, SystemView
from ....core.errors import ValidationError
from ....core.specs import Role
from ....core.systems import BaseSystem, System


class StockElement(Element):
    """Renewable stock `n` with the growth rate `r` and the carrying capacity `K`.

    `StockGrowthInteraction` grows it logistically.

    Args:
        label:
            The label of the stock.

        n:
            The initial stock.

        r:
            The intrinsic growth rate.

        K:
            The carrying capacity.

    Examples:
        >>> from skcomplex.systems.economics.resources import StockElement
        >>> StockElement('stock', n=500.0).n
        500.0
    """

    def __init__(
        self,
        label: str,
        *,
        n: Annotated[object, RealSet.closed_open(0, oo)] = 1000.0,
        r: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        K: Annotated[object, RealSet.open(0, oo)] = 1000.0,
    ) -> None:
        super().__init__(label)
        self.n = n
        self.r = r
        self.K = K


class StockGrowthInteraction(Interaction):
    """Logistic growth of a renewable stock, `n' = r n (1 - n / K)`.

    The rate `r` and the capacity `K` are the values of the stock.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the stock.

    Examples:
        >>> from skcomplex.systems.economics.resources import StockGrowthInteraction
        >>> StockGrowthInteraction('growth', target='stock').target
        'stock'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the logistic growth of the stock.

        Args:
            context:
                The namespace of the stock.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            stock = context.target
            self.laws_: list[BaseLaw] = [RateLaw(stock.n, stock.r * stock.n * (1 - stock.n / stock.K))]


class FarmerView(AgentView):
    """Farmer as its catch rule sees it: its `wealth`."""

    wealth: float


class FarmerElement(Element):
    """Farmer who adds the catch of every step to its wealth, by `CatchInteraction`.

    Args:
        label:
            The label of the agent or group.

        wealth:
            The initial wealth.

        values:
            Further values by name, or `None` for none.

        size:
            The number of farmers, or `None` for one.

        space:
            The space of the farmers, or `None` for none.

    Examples:
        >>> from skcomplex.systems.economics.resources import FarmerElement
        >>> FarmerElement('farmers', size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        *,
        wealth: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.wealth = wealth

    @classmethod
    def _get_view(cls) -> type[FarmerView]:
        """Return the view of the agents of the element."""
        return FarmerView


class CatchInteraction(Interaction):
    """Catch of every farmer, who adds the effort times the resource stock to its wealth.

    The farmers read the stock from another subsystem at its last synchronisation point.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of farmers.

        resource:
            The dotted name of the resource stock.

        effort:
            The dotted name of the harvest effort per farmer.

    Examples:
        >>> from skcomplex.systems.economics.resources import CatchInteraction
        >>> CatchInteraction('catch', target='farmers').resource
        'land/stock.n'
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        resource: str = 'land/stock.n',
        effort: str = 'harvest.effort',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.resource = resource
        self.effort = effort

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the catch rule of every farmer.

        Args:
            context:
                The namespace of the farmers.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = [UpdateLaw(context.target, self.catch)]

    def catch(self, agent: FarmerView, system: SystemView) -> None:
        """Add the effort times the resource stock to the wealth.

        The stock is read at the last synchronisation point.

        Args:
            agent:
                The farmer.

            system:
                The system.
        """
        stock = system.inputs.get(self.resource, 0.0)
        if not isinstance(stock, int | float | np.number):
            error_msg = f'The input {self.resource!r} of the farmers is {stock!r}, not a number.'
            raise ValidationError(error_msg)
        agent.wealth += system.parameters[self.effort] * float(stock)


class HarvestInteraction(Interaction):
    """Harvest of a stock proportional to the stock and to the number of harvesters.

    The rate is `-effort * N * n`, a rate on the state `n` of the target whose coefficient counts the members of the
    group of harvesters, which may live in another subsystem.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the harvested stock.

        harvesters:
            The path of the group of harvesters, which the harvest counts.

        effort:
            The harvest rate per harvester and unit of stock.

        state:
            The name of the state of the target that holds the stock.

    Examples:
        >>> from skcomplex.systems.economics.resources import HarvestInteraction
        >>> HarvestInteraction('harvest', target='land/stock', harvesters='village/farmers', effort=0.05).effort
        0.05
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        harvesters: Role,
        effort: Annotated[float, RealSet.closed_open(0, oo)] = 0.05,
        state: str = 'n',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.harvesters = harvesters
        self.effort = effort
        self.state = state

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the harvest `-effort * N * n` of the stock.

        Args:
            context:
                The namespace of the stock, the harvesters, and the effort.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            stock = getattr(context.target, self.state)
            self.laws_: list[BaseLaw] = [RateLaw(stock, -context.effort * count(context.harvesters) * stock)]


class HarvestingSystem(BaseSystem):
    """A socio-ecological system of farmer agents harvesting a resource.

    The resource regrows logistically. The resource settles at `K * (1 - effort * farmers / r)` when
    `effort * farmers < r`, and collapses otherwise. The system has the ODE subsystem `land` and the agent subsystem
    `village`.

    Args:
        farmers:
            The number of farmers.

        effort:
            The harvest rate per farmer and unit of stock.

        r:
            The growth rate of the resource.

        K:
            The carrying capacity of the resource.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.resources import HarvestingSystem
        >>> [subsystem.label for subsystem in HarvestingSystem().prepare().system_.systems]
        ['land', 'village']
    """

    field = 'economics/resources'
    topics = ('socio-ecological', 'agents', 'ode', 'co-simulation')
    references = ('Hamill and Gilbert 2016 ch. 10',)
    until = 50

    def __init__(
        self,
        farmers: int = 10,
        *,
        effort: float = 0.05,
        r: float = 1.0,
        K: float = 1000.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.farmers = farmers
        self.effort = effort
        self.r = r
        self.K = K

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        land = System(
            'land',
            elements=[StockElement('stock', n=self.K, r=self.r, K=self.K)],
            interactions=[StockGrowthInteraction('growth', target='stock')],
        )
        village = System(
            'village',
            elements=[FarmerElement('farmers', size=self.farmers)],
            interactions=[CatchInteraction('catch', target='farmers')],
        )
        harvest = HarvestInteraction('harvest', target='land/stock', harvesters='village/farmers', effort=self.effort)
        self.system_ = System(self.label, interactions=[harvest], systems=[land, village])
