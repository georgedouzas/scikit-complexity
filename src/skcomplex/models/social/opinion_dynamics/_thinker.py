"""Agent of the cognitive consistency model, with the valences of the concepts of its belief network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._thinker_view import ThinkerView


class ThinkerElement(Element):
    """Agent of the cognitive consistency model, with the valences of the concepts of its belief network.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        valences:
            The initial valences of the concepts F1, X, and F2, between -1 and 1.

        attitude:
            The attitude of the agent, the valence of the concept X once the belief network settles.

        decay:
            The fraction of every valence lost per update of the belief network.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.social.opinion_dynamics import ThinkerElement
        >>> group = GroupElement('thinkers', element=ThinkerElement, size=5, values={'valences': [1.0, 1.0, -1.0]})
        >>> group.label, group.element.__name__, group.size
        ('thinkers', 'ThinkerElement', 5)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'attitude', 'color_range': (-1.0, 1.0)}),
        *,
        valences: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
        attitude: Annotated[float, StateSpec()] = 0.0,
        decay: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.valences = valences
        self.attitude = attitude
        self.decay = decay

    @classmethod
    def _get_view(cls) -> type[ThinkerView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return ThinkerView
