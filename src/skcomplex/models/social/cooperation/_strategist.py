"""Agent of the prisoner's dilemma tournament, with a genome of responses and the fitness of its last games."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec
from ._strategist_view import StrategistView


class StrategistElement(Element):
    """Agent of the prisoner's dilemma tournament, with a genome of responses and the fitness of its last games.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        genome:
            The seven moves, 1 to cooperate and 0 to defect, for the opponent's last two moves: none, C, D, CC, CD,
            DC, and DD.

        fitness:
            The mean score per round of the last tournament.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.social.cooperation import StrategistElement
        >>> group = GroupElement('agents', element=StrategistElement, size=10, values={'genome': [1.0] * 7})
        >>> group.label, group.element.__name__, group.size
        ('agents', 'StrategistElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        genome: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 1.0,
        fitness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.genome = genome
        self.fitness = fitness

    @classmethod
    def _get_view(cls) -> type[StrategistView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return StrategistView
