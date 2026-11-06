"""Shopper who visits one stall a step that it has not visited yet and buys what it wants there."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import SystemView
from ....core.specs import StateSpec, VisualSpec
from ._shopper_view import ShopperView
from ._stall_view import StallView

WHOLESALE_PRICES = (0.6, 0.5, 0.7, 0.9, 1.2, 1.5, 0.8, 0.4, 0.5, 0.3, 0.6, 1.0)


class ShopperElement(Element):
    """Shopper who visits one stall a step that it has not visited yet and buys what it wants there.

    A shopper gives up the items that it has not found once it has visited every stall.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        wanted:
            The bit mask of the items on the shopping list.

        visited:
            The bit mask of the stalls already visited.

        missed:
            The bit mask of the items given up.

        spent:
            The money spent.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import ShopperElement
        >>> GroupElement('shoppers', element=ShopperElement, size=5, values={'wanted': 3}).size
        5
    """

    stalls: ClassVar[str] = 'stalls'

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        wanted: Annotated[float, StateSpec(domain=NN)] = 0,
        visited: Annotated[float, StateSpec(domain=NN)] = 0,
        missed: Annotated[float, StateSpec(domain=NN)] = 0,
        spent: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.wanted = wanted
        self.visited = visited
        self.missed = missed
        self.spent = spent

    @classmethod
    def _get_view(cls) -> type[ShopperView]:
        """Return the view of a shopper, with its list, visits, misses, and spending."""
        return ShopperView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the shopping rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.shop)]

    def shop(self, agent: ShopperView, system: SystemView) -> None:
        """Visit a random stall not visited yet and buy every wanted item that it sells.

        Args:
            agent:
                The shopper.

            system:
                The system.
        """
        wanted, visited = int(agent.wanted), int(agent.visited)
        if not wanted:
            return
        unvisited = [stall for stall in system.agents(self.stalls, StallView) if not visited >> int(stall.stall) & 1]
        if not unvisited:
            agent.missed, agent.wanted = wanted, 0
            return
        stall = unvisited[int(system.rng.integers(len(unvisited)))]
        bought = wanted & int(stall.stock)
        agent.visited = visited | 1 << int(stall.stall)
        agent.spent += (1 + stall.markup) * sum(WHOLESALE_PRICES[item] for item in ShopperElement._items_of(bought))
        agent.wanted = wanted & ~bought

    @staticmethod
    def _items_of(mask: int) -> list[int]:
        """Return the indices of the items in a bit mask."""
        return [item for item in range(len(WHOLESALE_PRICES)) if mask >> item & 1]
