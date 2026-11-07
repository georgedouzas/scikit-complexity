"""Natural resources systems: harvesting renewable stocks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import RateInteraction
from ....core.components.laws import BaseLaw, RateLaw, RuleLaw
from ....core.components.laws.utils import count
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec


class StockElement(Element):
    """Renewable stock that grows logistically, `n' = r n (1 - n / K)`.

    Args:
        label:
            The label of the stock.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        r: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        K: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1000.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n
        self.r = r
        self.K = K

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the logistic growth of the stock.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.n, s.r * s.n * (1 - s.n / s.K))]


class FarmerView(AgentView):
    """Farmer as its catch rule sees it: its `wealth`."""

    wealth: float


class FarmerElement(Element):
    """Farmer who adds the catch of every step to its wealth, reading the resource from the other partition.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        wealth:
            The initial wealth.

        resource:
            The dotted name of the resource stock.

        effort:
            The dotted name of the harvest effort per farmer.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.resources import FarmerElement
        >>> GroupElement('farmers', element=FarmerElement, size=10).size
        10
    """

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        wealth: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        resource: str = 'land/stock.n',
        effort: str = 'harvest.effort',
    ) -> None:
        super().__init__(label, visual=visual)
        self.wealth = wealth
        self.resource = resource
        self.effort = effort

    @classmethod
    def _get_view(cls) -> type[FarmerView]:
        """Return the view of the agents of the element."""
        return FarmerView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the catch rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.catch)]

    def catch(self, agent: FarmerView, system: SystemView) -> None:
        """Add the effort times the resource stock at the last synchronisation point to the wealth.

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


class HarvestInteraction(RateInteraction):
    """Harvest of a stock proportional to the stock and to the number of harvesters, the rate `-effort * N * n`.

    A rate on the state `n` of the target whose coefficient counts the members of the group of harvesters, which may
    live in another subsystem.

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

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        harvesters: Annotated[str, RoleSpec()],
        effort: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        state: str = 'n',
    ) -> None:
        super().__init__(label, target=target, state=state)
        self.harvesters = harvesters
        self.effort = effort

    def _get_rate(self, namespace: SimpleNamespace) -> object:
        """Return the harvest `-effort * N * n`.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the target and the harvesters.

        Returns:
            rate:
                The harvest rate of the stock.
        """
        return -namespace.effort * count(namespace.harvesters) * getattr(namespace.target, self.state)


class HarvestingSystem(BaseSystem):
    """A socio-ecological system of farmer agents harvesting a logistically regrowing resource.

    The resource settles at `K * (1 - effort * farmers / r)` when `effort * farmers < r`, and collapses otherwise. The
    system has the ODE subsystem `land` and the agent subsystem `village`.

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
    ) -> None:
        super().__init__(label)
        self.farmers = farmers
        self.effort = effort
        self.r = r
        self.K = K

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        land = System('land', elements=[StockElement('stock', n=self.K, r=self.r, K=self.K)])
        village = System('village', elements=[GroupElement('farmers', element=FarmerElement, size=self.farmers)])
        harvest = HarvestInteraction('harvest', target='land/stock', harvesters='village/farmers', effort=self.effort)
        return System(self.label, interactions=[harvest], systems=[land, village])
