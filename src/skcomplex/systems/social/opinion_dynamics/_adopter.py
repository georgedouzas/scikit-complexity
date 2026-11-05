"""Agent of the threshold adoption system, who adopts an innovation for good."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._adopter_view import AdopterView


class AdopterElement(Element):
    """Agent of the threshold adoption system, who adopts an innovation for good.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        adopted:
            Whether the agent starts as an adopter.

        threshold:
            The percentage of adopters in the agent's social circle above which it adopts in the influence mode.

        reach:
            The distance within which other agents form an agent's social circle.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.opinion_dynamics import AdopterElement
        >>> group = GroupElement('people', element=AdopterElement, size=10, values={'threshold': 50.0})
        >>> group.label, group.element.__name__, group.size
        ('people', 'AdopterElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'adopted', 'color_range': (0.0, 1.0)}),
        *,
        adopted: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        threshold: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 50.0,
        reach: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 15.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.adopted = adopted
        self.threshold = threshold
        self.reach = reach

    @classmethod
    def _get_view(cls) -> type[AdopterView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return AdopterView
