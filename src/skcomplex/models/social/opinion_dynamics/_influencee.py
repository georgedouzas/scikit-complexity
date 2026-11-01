"""Agent of the social impact model, with an opinion of -1 or 1, a persuasiveness, and a supportiveness."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import ZZ, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._influencee_view import InfluenceeView


class InfluenceeElement(Element):
    """Agent of the social impact model, with an opinion of -1 or 1, a persuasiveness, and a supportiveness.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The initial opinion, -1 or 1.

        persuasiveness:
            The strength with which the agent persuades agents of the other opinion.

        supportiveness:
            The strength with which the agent supports agents of its opinion.

        exponent:
            The exponent of the distance by which impact decays.

        noise:
            The half-width of the uniform random field added to the impact on every agent and step.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.social.opinion_dynamics import InfluenceeElement
        >>> group = GroupElement('people', element=InfluenceeElement, size=4, values={'opinion': -1.0})
        >>> group.label, group.element.__name__, group.size
        ('people', 'InfluenceeElement', 4)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (-1.0, 1.0)}),
        *,
        opinion: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
        persuasiveness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        supportiveness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        exponent: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        noise: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.opinion = opinion
        self.persuasiveness = persuasiveness
        self.supportiveness = supportiveness
        self.exponent = exponent
        self.noise = noise

    @classmethod
    def _get_view(cls) -> type[InfluenceeView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return InfluenceeView
