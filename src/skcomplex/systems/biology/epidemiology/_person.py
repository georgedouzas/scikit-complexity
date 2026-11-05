"""Person of the agent-based SIR system, with incubation, mortality, waning immunity, and lockdown."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._person_view import PersonView

SUSCEPTIBLE, EXPOSED, INFECTED, RECOVERED, DEAD = 0.0, 1.0, 2.0, 3.0, 4.0


class PersonElement(Element):
    """Person of the agent-based SIR system, with incubation, mortality, waning immunity, and lockdown.

    A `MovementInteraction` walks a living person to a neighbouring cell with probability `d`, or `lockdown_d` during
    the lockdown, which lasts `lockdown_length` steps from `lockdown_start`. Transitions of the agent form, which
    `agent_based_sir` adds, move the health of the person. An infectious person infects, with probability `beta`,
    every susceptible person in its cell, who becomes exposed. An infectious person recovers with probability `gamma`
    or dies with probability `mu`, and a recovered person loses immunity with probability `alpha`. Health is 0
    susceptible, 1 exposed, 2 infected, 3 recovered, or 4 dead.

    The incubation follows the convention of the residence time `after` of `TransitionInteraction`: the step of the
    exposure counts as the first step in the exposed state, so a person exposed in step `n` becomes infectious at the
    end of step `n + incubation - 1`. The clock that the person kept before counted the step of the exposure only when
    the person acted after its infector in that step, so the onset now comes half a step earlier on average.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        health:
            The health state.

        d:
            The probability of moving at every step.

        beta:
            The probability that an infectious person infects the susceptible persons of its cell.

        gamma:
            The probability of recovering at every step.

        mu:
            The probability of dying at every step of the infection.

        alpha:
            The probability of losing immunity at every step.

        lockdown_start:
            The step at which the lockdown starts.

        lockdown_length:
            The number of steps of the lockdown, 0 for none.

        lockdown_d:
            The probability of moving during the lockdown.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.epidemiology import PersonElement
        >>> GroupElement('people', element=PersonElement, size=100).size
        100
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'health', 'color_range': (0.0, 4.0)}),
        *,
        health: Annotated[float, StateSpec(domain=NN)] = SUSCEPTIBLE,
        d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.6,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        mu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        lockdown_start: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        lockdown_length: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        lockdown_d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.health = health
        self.d = d
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha
        self.lockdown_start = lockdown_start
        self.lockdown_length = lockdown_length
        self.lockdown_d = lockdown_d

    @classmethod
    def _get_view(cls) -> type[PersonView]:
        """Return the view of a person, `PersonView`."""
        return PersonView
