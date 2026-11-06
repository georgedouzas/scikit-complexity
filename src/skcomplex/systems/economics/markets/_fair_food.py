"""Market systems: consumers choosing between fair-trade and regular food under ethics, budget, and peer influence."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import ChoiceInteraction
from ....core.components.spaces import DiscreteTime, LatticeSpace
from ....core.components.system import System
from ....core.engine.solvers import SystemView
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system
from ._consumer_view import ConsumerView


def _get_utility(agent: ConsumerView, system: SystemView, option: float) -> float:
    """Return the utility of fair food, option 1, or regular food, option 0, `-inf` for fair food over the budget."""
    if option == 0:
        return 0.0
    if agent.budget < 1 + agent.premium:
        return -math.inf
    neighbors = [neighbor for neighbor in agent.neighbors() if isinstance(neighbor, ConsumerView)]
    share = sum(neighbor.fair for neighbor in neighbors) / len(neighbors) if neighbors else 0.0
    return float(
        agent.ethics_weight * agent.ethics
        - agent.price_weight * agent.premium / agent.budget
        + agent.social_weight * share,
    )


class ConsumerElement(Element):
    """Consumer who buys fair-trade or regular food, with an ethical concern, a budget, and the weights of its utility.

    The utility of fair food is `ethics_weight * ethics - price_weight * premium / budget + social_weight * share`,
    where `share` is the share of neighbours who bought fair food in the last step, and the utility of regular food is
    zero. A consumer whose budget does not cover the price `1 + premium` of fair food buys regular food.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        fair:
            One when the consumer bought fair food in the last step, zero otherwise.

        ethics:
            The ethical concern of the consumer, between zero and one.

        budget:
            The money the consumer can spend on one unit of food, regular food costing one.

        premium:
            The extra price of fair food over regular food.

        ethics_weight:
            The weight of ethical concern in the utility of fair food.

        price_weight:
            The weight of the premium relative to the budget.

        social_weight:
            The weight of the share of neighbours who buy fair food.

    Examples:
        >>> from skcomplex.systems.economics.markets import ConsumerElement
        >>> GroupElement('consumers', element=ConsumerElement, size=4, values={'ethics': 0.5}).size
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'fair', 'color_range': (0.0, 1.0)}),
        *,
        fair: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        ethics: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        budget: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        premium: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        ethics_weight: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        price_weight: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        social_weight: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.fair = fair
        self.ethics = ethics
        self.budget = budget
        self.premium = premium
        self.ethics_weight = ethics_weight
        self.price_weight = price_weight
        self.social_weight = social_weight

    @classmethod
    def _get_view(cls) -> type[ConsumerView]:
        """Return the view of a consumer, with its states and the parameters of its utility."""
        return ConsumerView


@register_system(
    field='economics/markets',
    tags=('consumer choice', 'social influence', 'agents', 'grid'),
    references=('Chesney 2021 ch. 3', 'Chesney 2021 ch. 7'),
    defaults={'until': 50},
    observables=lambda system: [ObservableSpec('fair_share', group='consumers', variable='fair')],
)
def fair_food(
    width: int = 20,
    height: int = 20,
    *,
    premium: float = 0.3,
    ethics_weight: float = 2.0,
    price_weight: float = 5.0,
    social_weight: float = 2.0,
    intensity: float = 1.0,
    low_budget: float = 1.0,
    high_budget: float = 3.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Chesney's choice between fair-trade and regular food of consumers who influence their neighbours.

    Every consumer, of uniform ethical concern and budget, sits on a cell of a grid and every step buys fair or
    regular food, all at once from the choices of their neighbours in the last step, as in the choice of Ajani. The
    share of fair food rises with the weight of ethics and falls with the premium, and social influence raises it.

    Args:
        width:
            The number of columns of the grid.

        height:
            The number of rows of the grid.

        premium:
            The extra price of fair food over regular food, which costs one.

        ethics_weight:
            The weight of ethical concern in the utility of fair food.

        price_weight:
            The weight of the premium relative to the budget.

        social_weight:
            The weight of the share of neighbours who buy fair food.

        intensity:
            The intensity of the logit choice.

        low_budget:
            The lowest budget.

        high_budget:
            The highest budget.

        random_state:
            The seed or generator of the ethics and the budgets.

    Returns:
        system:
            The system, with the group `consumers`, their logit choice `shopping`, and the share of fair food
            `fair_share`.

    Examples:
        >>> from skcomplex.systems.economics.markets import fair_food
        >>> fair_food(5, 4).elements[0].prepare().size_
        20
    """
    rng = np.random.default_rng(random_state)
    size = width * height
    consumers = GroupElement(
        'consumers',
        element=ConsumerElement,
        size=size,
        values={
            'ethics': rng.random(size),
            'budget': rng.uniform(low_budget, high_budget, size),
            'premium': premium,
            'ethics_weight': ethics_weight,
            'price_weight': price_weight,
            'social_weight': social_weight,
        },
        location=[(x, y) for x in range(width) for y in range(height)],
    )
    choice = ChoiceInteraction(
        'shopping',
        target='consumers',
        state='fair',
        options=(0, 1),
        utility=_get_utility,
        intensity=intensity,
        form='agent',
    )
    return System(
        elements=[consumers],
        interactions=[choice],
        space=LatticeSpace((width, height), boundary='periodic', capacity=1),
        time=DiscreteTime(),
    )
