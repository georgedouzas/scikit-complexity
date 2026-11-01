"""Agent of the Hegselmann-Krause model, with an opinion and the confidence within which it trusts neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class ConfidentElement(Element):
    """Agent of the Hegselmann-Krause model, with an opinion and the confidence within which it trusts neighbours.

    `ConfidenceCouplingInteraction` moves every agent each step to the mean opinion of the neighbours, itself
    included, whose opinions differ from its own by less than `confidence`.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The opinion, between 0 and 1.

        confidence:
            The largest opinion difference of a neighbour an agent still listens to.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import ConfidentElement
        >>> ConfidentElement('people', confidence=0.1).confidence
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('histograms', {'color_range': (0.0, 1.0), 'histogram_first': True}),
        *,
        opinion: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        confidence: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.opinion = opinion
        self.confidence = confidence
