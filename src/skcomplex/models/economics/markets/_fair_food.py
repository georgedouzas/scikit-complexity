"""Markets models: consumers choosing between fair-trade and regular food under ethics, budget, and social influence."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np

from ....core.base import Domain, VisualSpec, parameter, state
from ....core.components import Element
from ....core.laws import Law, Rule, Symbols
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView
from ....core.space import DiscreteTime, LatticeSpace, Phase, Phases
from ....core.system import Observable, System
from ..realizations import logit_choice


class Consumer(Element):
    """Consumer who buys fair-trade or regular food by a logit choice on the utility of each option.

    The utility of fair food is `ethics_weight * ethics - price_weight * premium / budget + social_weight * share`,
    where `share` is the share of neighbours who bought fair food in the last step, and the utility of regular food is
    zero. A consumer whose budget does not cover the price `1 + premium` of fair food buys regular food.

    Args:
        label:
            The label of the agent or group.

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

        intensity:
            The intensity of the logit choice.

    Examples:
        >>> from skcomplex.models.economics.markets import Consumer
        >>> Consumer.group('consumers', 4, ethics=0.5).size
        4
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'fair', 'color_range': (0.0, 1.0)})
    fair: float = state(default=0.0, domain=Domain.BOOLEAN)
    ethics: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    budget: float = state(default=2.0, domain=Domain.NONNEGATIVE)
    premium: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)
    ethics_weight: float = parameter(default=2.0, domain=Domain.NONNEGATIVE)
    price_weight: float = parameter(default=5.0, domain=Domain.NONNEGATIVE)
    social_weight: float = parameter(default=2.0, domain=Domain.NONNEGATIVE)
    intensity: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the shopping rule in the phase of shopping.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.shop, phase='shopping')]

    def shop(self, agent: AgentView, model: ModelView) -> None:
        """Buy fair or regular food with logit probabilities of their utilities.

        Args:
            agent:
                The consumer.

            model:
                The model.
        """
        if agent.budget < 1 + agent.premium:
            agent.fair = 0.0
            return
        neighbors = agent.neighbors()
        share = sum(neighbor.fair for neighbor in neighbors) / len(neighbors) if neighbors else 0.0
        utility = (
            agent.ethics_weight * agent.ethics
            - agent.price_weight * agent.premium / agent.budget
            + agent.social_weight * share
        )
        agent.fair = float(logit_choice([0.0, utility], agent.intensity, model.rng))


@register_model(
    field='economics/markets',
    tags=('consumer choice', 'social influence', 'agents', 'grid'),
    references=('Chesney 2021 ch. 3', 'Chesney 2021 ch. 7'),
    defaults={'until': 50},
    observables=lambda system: [Observable('fair_share', group='consumers', variable='fair')],
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
            The model, with the group `consumers` and the share of fair food `fair_share`.

    Examples:
        >>> from skcomplex.models.economics.markets import fair_food
        >>> fair_food(5, 4).elements[0].size
        20
    """
    rng = np.random.default_rng(random_state)
    size = width * height
    consumers = Consumer.group(
        'consumers',
        size,
        location=[(x, y) for x in range(width) for y in range(height)],
        ethics=rng.random(size),
        budget=rng.uniform(low_budget, high_budget, size),
        premium=premium,
        ethics_weight=ethics_weight,
        price_weight=price_weight,
        social_weight=social_weight,
        intensity=intensity,
    )
    return System(
        components=[consumers],
        space=LatticeSpace((width, height), boundary='periodic', capacity=1),
        time=DiscreteTime(schedule=Phases([Phase('shopping', activation='simultaneous')])),
    )
