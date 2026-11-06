"""Agent that is in a club (spin 1) or out of it (spin 0), aligned by a heat bath with the agents it knows."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._coevolving_voter import OPINION_VISUAL


class ClubMemberElement(Element):
    """Agent that is in a club (spin 1) or out of it (spin 0), aligned by a heat bath with the agents it knows.

    Every step an agent reconsiders with probability `deciding` and then joins with probability
    `1 / (1 + exp(-2 β h))`, where the field `h` counts the acquaintances in the club minus those out of it, plus the
    appeal of the club. The interaction `SeparationInteraction` dissolves acquaintances between a member and an outsider
    and reforms them towards an agent of the same side.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            1 for a member of the club and 0 for an outsider.

        degree:
            The number of acquaintances, read at the start of every step.

        beta:
            The inverse temperature of the heat bath.

        appeal:
            The field of the club on every agent, positive for an attractive club.

        deciding:
            The probability that an agent reconsiders its choice in a step.

        activity:
            The probability that an acquaintance is reconsidered in a step.

        rewiring:
            The probability that a reconsidered acquaintance across the club boundary dissolves and reforms.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import ClubMemberElement
        >>> ClubMemberElement('agents', beta=2.0).beta
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = OPINION_VISUAL,
        *,
        opinion: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        degree: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        appeal: Annotated[float, ParameterSpec()] = 0.0,
        deciding: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        activity: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        rewiring: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.opinion = opinion
        self.degree = degree
        self.beta = beta
        self.appeal = appeal
        self.deciding = deciding
        self.activity = activity
        self.rewiring = rewiring
