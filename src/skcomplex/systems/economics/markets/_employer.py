"""Firm that posts a vacancy with a random wage offer in some steps, filled by at most one worker."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.engine.solvers import SystemView
from ....core.specs import (
    ParameterSpec,
    StateSpec,
    VisualSpec,
)
from ._employer_view import EmployerView


class EmployerElement(Element):
    """Firm that posts a vacancy with a random wage offer in some steps, filled by at most one worker.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        bid:
            The wage offered for the vacancy.

        demand:
            The number of open vacancies, one or zero.

        posted:
            The step in which the vacancy was posted.

        vacancy_rate:
            The probability of posting a vacancy in a step.

        low_wage:
            The lowest wage offer.

        high_wage:
            The highest wage offer.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import EmployerElement
        >>> GroupElement('firms', element=EmployerElement, size=5).size
        5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        bid: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        demand: Annotated[float, StateSpec(domain=NN)] = 0.0,
        posted: Annotated[float, StateSpec()] = -1.0,
        vacancy_rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        low_wage: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        high_wage: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.bid = bid
        self.demand = demand
        self.posted = posted
        self.vacancy_rate = vacancy_rate
        self.low_wage = low_wage
        self.high_wage = high_wage

    @classmethod
    def _get_view(cls) -> type[EmployerView]:
        """Return the view of an employer, with its vacancy and the parameters of its offers."""
        return EmployerView

    @staticmethod
    def get_posting() -> tuple[tuple[str, Callable[[EmployerView, SystemView], float]], ...]:
        """Return the states a firm sets in the phase of offers, in their order.

        The firm withdraws its last vacancy and posts a new one at the vacancy rate, with a uniform wage offer and the
        step of posting.

        Returns:
            posting:
                The name of every state and the function of its value, of the firm and the system view.
        """
        return (
            ('demand', lambda agent, system: float(system.rng.random() < agent.vacancy_rate)),
            (
                'bid',
                lambda agent, system: (
                    float(system.rng.uniform(agent.low_wage, agent.high_wage)) if agent.demand else agent.bid
                ),
            ),
            ('posted', lambda agent, system: system.time if agent.demand else agent.posted),
        )
