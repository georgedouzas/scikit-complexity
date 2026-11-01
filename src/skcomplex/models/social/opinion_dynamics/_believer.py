"""Agent of the belief diffusion model, with the weights it gives to four levels of caring about an issue."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._believer_view import BelieverView


class BelieverElement(Element):
    """Agent of the belief diffusion model, with the weights it gives to four levels of caring about an issue.

    The levels are not caring at all, caring less, caring a bit, and caring passionately. An agent cares at the level
    of its largest weight. In `influentials`, a `MapInteraction` of the vector of weights adds every step `learning`
    times the influence of every neighbour to the weight of the level the neighbour cares at, a tie broken at random,
    and a second one sets the level of caring.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        belief:
            The initial weights of the four levels.

        caring:
            The initial level of caring, the index of the largest weight.

        influence:
            The weight of the agent's own level in the beliefs of its neighbours.

        learning:
            The increment per neighbour and step.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.social.opinion_dynamics import BelieverElement
        >>> group = GroupElement('people', element=BelieverElement, size=4, values={'belief': [1.0, 0.0, 0.0, 0.0]})
        >>> group.label, group.element.__name__, group.size
        ('people', 'BelieverElement', 4)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'caring', 'color_range': (0.0, 3.0)}),
        *,
        belief: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
        caring: Annotated[float, StateSpec(domain=NN)] = 0.0,
        influence: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        learning: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.belief = belief
        self.caring = caring
        self.influence = influence
        self.learning = learning

    @classmethod
    def _get_view(cls) -> type[BelieverView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return BelieverView
