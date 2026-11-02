"""Worker who loses its job at the separation rate and, while unemployed, lowers its reservation wage."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....engine.solvers import ModelView
from ....specs import (
    ParameterSpec,
    StateSpec,
    VisualSpec,
)
from ._jobseeker_view import JobseekerView


class JobseekerElement(Element):
    """Worker who loses its job at the separation rate and, while unemployed, lowers its reservation wage.

    A worker who loses its job searches from the next step. An unemployed worker asks the reservation wage
    `w_0 (1 - decline)^d`, where `d` is the number of steps of search, and is hired by the matching market.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        ask:
            The reservation wage.

        supply:
            One while the worker searches in the current step, zero otherwise.

        employer:
            The id of the employer, or -1 for an unemployed worker.

        wage:
            The wage of the current job.

        duration:
            The number of steps of the current spell of search.

        hired:
            The step of the last hire.

        offer:
            The step in which the offer of the last hire was posted.

        separation:
            The probability that a job is destroyed in a step.

        reservation:
            The reservation wage at the start of a search.

        decline:
            The relative fall of the reservation wage per step of search.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.markets import JobseekerElement
        >>> GroupElement('workers', element=JobseekerElement, size=5).size
        5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        ask: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        supply: Annotated[float, StateSpec(domain=NN)] = 1.0,
        employer: Annotated[float, StateSpec(reference='firms')] = -1.0,
        wage: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        duration: Annotated[float, StateSpec(domain=NN)] = 0.0,
        hired: Annotated[float, StateSpec()] = -1.0,
        offer: Annotated[float, StateSpec()] = -1.0,
        separation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        reservation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        decline: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.ask = ask
        self.supply = supply
        self.employer = employer
        self.wage = wage
        self.duration = duration
        self.hired = hired
        self.offer = offer
        self.separation = separation
        self.reservation = reservation
        self.decline = decline

    @classmethod
    def _get_view(cls) -> type[JobseekerView]:
        """Return the view of a worker, with its search and job states and their parameters."""
        return JobseekerView

    @staticmethod
    def get_search() -> tuple[tuple[str, Callable[[JobseekerView, ModelView], float]], ...]:
        """Return the states a worker sets in the phase of offers before the separations, in their order.

        A worker unemployed at the start of the step searches, with a reservation wage that falls with the duration of
        its search.

        Returns:
            search:
                The name of every state and the function of its value, of the worker and the model view.
        """
        return (
            ('supply', lambda agent, model: float(agent.employer < 0)),
            (
                'ask',
                lambda agent, model: (
                    agent.reservation * (1 - agent.decline) ** agent.duration if agent.supply else agent.ask
                ),
            ),
        )

    @staticmethod
    def get_spell() -> tuple[tuple[str, Callable[[JobseekerView, ModelView], float]], ...]:
        """Return the states a worker sets in the phase of offers after the separations, in their order.

        A worker who lost its job earns no wage, and the duration of a search grows while the worker searches.

        Returns:
            spell:
                The name of every state and the function of its value, of the worker and the model view.
        """
        return (
            ('wage', lambda agent, model: 0.0 if agent.employer < 0 else agent.wage),
            (
                'duration',
                lambda agent, model: (
                    agent.duration + 1 if agent.supply else 0.0 if agent.employer < 0 else agent.duration
                ),
            ),
        )
