"""Distinct word of the Simon system, with the number of times it occurs in the text."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._growing_word_view import GrowingWordView


class GrowingWordElement(Element):
    """Distinct word of the Simon system, with the number of times it occurs in the text.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        count:
            The number of occurrences.

        innovation:
            The probability that the next word of the text is new.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.spreading import GrowingWordElement
        >>> group = GroupElement('words', element=GrowingWordElement, size=1, values={'count': 1.0})
        >>> group.label, group.element.__name__, group.size
        ('words', 'GrowingWordElement', 1)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        count: Annotated[float, StateSpec(domain=NN)] = 1.0,
        innovation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.count = count
        self.innovation = innovation

    @classmethod
    def _get_view(cls) -> type[GrowingWordView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return GrowingWordView
