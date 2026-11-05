"""Customer who remembers how satisfied it was with the products of each of two shops."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._customer_view import CustomerView


class CustomerElement(Element):
    """Customer who remembers how satisfied it was with the products of each of two shops.

    The customer visits the shop `k` of larger satisfaction `S_k` more often, with the logit utility `S_k / T`, and
    holds the freshness of the product it bought last.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        s1:
            The satisfaction with the first shop.

        s2:
            The satisfaction with the second shop.

        shop:
            The index of the shop the customer visits, 0 or 1.

        freshness:
            The freshness of the product the customer bought last.

        T:
            The temperature of the choice, the randomness of the customer.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import CustomerElement
        >>> GroupElement('customers', element=CustomerElement, size=10, values={'s1': 0.5, 's2': 0.5}).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        s1: Annotated[float, StateSpec()] = 0.5,
        s2: Annotated[float, StateSpec()] = 0.5,
        shop: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        freshness: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 1.0,
        T: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.s1 = s1
        self.s2 = s2
        self.shop = shop
        self.freshness = freshness
        self.T = T

    @classmethod
    def _get_view(cls) -> type[CustomerView]:
        """Return the view of a customer, with its satisfactions, shop, freshness, and temperature."""
        return CustomerView

    @staticmethod
    def utility(agent: CustomerView, system: SystemView, option: float) -> float:
        """Return the utility `S_k / T` of visiting shop `k`.

        Args:
            agent:
                The customer.

            system:
                The system view of the choice.

            option:
                The index of the shop.

        Returns:
            utility:
                The satisfaction with the shop divided by the temperature.
        """
        return (agent.s1 if option == 0 else agent.s2) / agent.T

    @staticmethod
    def destination(agent: AgentViewProtocol, system: SystemViewProtocol) -> int | None:
        """Return the node of the shop the customer chose.

        Args:
            agent:
                The customer.

            system:
                The system view of the movement.

        Returns:
            node:
                The node of the shop, or `None` to stay for an agent that is not a customer.
        """
        return int(agent.shop) if isinstance(agent, CustomerView) else None

    @staticmethod
    def payoff(agent: CustomerView, system: SystemView, option: int) -> float:
        """Return the freshness of the product the customer bought last, the payoff of the shop it visited.

        Args:
            agent:
                The customer.

            system:
                The system view of the reinforcement.

            option:
                The index of the shop the customer visited.

        Returns:
            freshness:
                The freshness of the product.
        """
        return agent.freshness

    @staticmethod
    def preference(agent: CustomerView) -> float:
        """Return `p_1 - p_2 = tanh((S_1 - S_2) / 2T)`, how much more likely the customer visits the first shop.

        Args:
            agent:
                The customer.

        Returns:
            preference:
                The difference of the probabilities of visiting the first and the second shop.
        """
        return float(np.tanh((agent.s1 - agent.s2) / (2 * agent.T)))
