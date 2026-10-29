"""Natural resources models: harvesting renewable stocks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RateLaw, RuleLaw
from ....core.laws.utils import count
from ....core.system import System
from ....engine.solvers import ModelView
from ....errors import ValidationError
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ._farmer_view import FarmerView


class StockElement(Element):
    """Stock element."""

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
        """Return the laws of the stock element."""
        return [RateLaw(s.n, s.r * s.n * (1 - s.n / s.K))]


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
        >>> from skcomplex.models.economics.resources import FarmerElement
        >>> GroupElement('farmers', element=FarmerElement, size=10).size
        10
    """

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

    def catch(self, agent: FarmerView, model: ModelView) -> None:
        """Add the effort times the resource stock at the last synchronisation point to the wealth.

        Args:
            agent:
                The farmer.

            model:
                The model.
        """
        stock = model.inputs.get(self.resource, 0.0)
        if not isinstance(stock, int | float | np.number):
            error_msg = f'The input {self.resource!r} of the farmers is {stock!r}, not a number.'
            raise ValidationError(error_msg)
        agent.wealth += model.parameters[self.effort] * float(stock)


class HarvestInteraction(Interaction):
    """Harvest of a resource proportional to its stock and to the number of harvesters, `-effort * N * n`.

    Args:
        label:
            The label of the interaction.

        resource:
            The path of the resource stock.

        harvesters:
            The path of the group of harvesters, which the harvest counts.

        effort:
            The harvest rate per harvester and unit of stock.

    Examples:
        >>> from skcomplex.models.economics.resources import HarvestInteraction
        >>> HarvestInteraction('harvest', resource='land/stock', harvesters='village/farmers', effort=0.05).effort
        0.05
    """

    def __init__(
        self,
        label: str,
        *,
        resource: Annotated[str, RoleSpec()],
        harvesters: Annotated[str, RoleSpec()],
        effort: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label)
        self.resource = resource
        self.harvesters = harvesters
        self.effort = effort

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the harvest rate.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.resource.n, -s.effort * count(s.harvesters) * s.resource.n)]


@register_model(
    field='economics/resources',
    tags=('socio-ecological', 'agents', 'ode', 'co-simulation'),
    references=('Hamill and Gilbert 2016 ch. 10',),
    defaults={'until': 50},
)
def harvesting(farmers: int = 10, *, effort: float = 0.05, r: float = 1.0, K: float = 1000.0) -> System:
    """Return a socio-ecological model of farmer agents harvesting a logistically regrowing resource.

    The resource settles at `K * (1 - effort * farmers / r)` when `effort * farmers < r`, and collapses otherwise.

    Args:
        farmers:
            The number of farmers.

        effort:
            The harvest rate per farmer and unit of stock.

        r:
            The growth rate of the resource.

        K:
            The carrying capacity of the resource.

    Returns:
        system:
            The model, with the ODE subsystem `land` and the agent subsystem `village`.

    Examples:
        >>> from skcomplex.models.economics.resources import harvesting
        >>> [subsystem.label for subsystem in harvesting().systems]
        ['land', 'village']
    """
    land = System('land', elements=[StockElement('stock', n=K, r=r, K=K)])
    village = System('village', elements=[GroupElement('farmers', element=FarmerElement, size=farmers)])
    harvest = HarvestInteraction('harvest', resource='land/stock', harvesters='village/farmers', effort=effort)
    return System(interactions=[harvest], systems=[land, village])
