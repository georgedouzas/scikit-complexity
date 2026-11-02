"""Agent of the Deffuant model, with a continuous opinion and a confidence bound."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._discussant_view import DiscussantView


class DiscussantElement(Element):
    """Agent of the Deffuant model, with a continuous opinion and a confidence bound.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The initial opinion, between 0 and 1.

        confidence:
            The largest opinion difference at which two agents still compromise.

        convergence:
            The fraction of the opinion difference by which each agent of a compromising pair moves.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.social.opinion_dynamics import DiscussantElement
        >>> group = GroupElement('people', element=DiscussantElement, size=10, values={'opinion': 0.5})
        >>> group.label, group.element.__name__, group.size
        ('people', 'DiscussantElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (0.0, 1.0)}),
        *,
        opinion: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        confidence: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        convergence: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.opinion = opinion
        self.confidence = confidence
        self.convergence = convergence

    @classmethod
    def _get_view(cls) -> type[DiscussantView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return DiscussantView
