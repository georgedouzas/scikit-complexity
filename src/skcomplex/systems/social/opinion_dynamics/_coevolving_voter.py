"""Voter of a coevolving network, with a binary opinion, its degree, and its number of discordant neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

OPINION_VISUAL = VisualSpec('sites', {'states': ['opinion'], 'low': 0.0, 'high': 1.0})


class CoevolvingVoterElement(Element):
    """Voter of a coevolving network, with a binary opinion, its degree, and its number of discordant neighbours.

    Every step a voter is active with probability `activity` and picks a random neighbour. When the neighbour
    disagrees, the voter copies its opinion with probability `1 - rewiring`, and otherwise the interaction
    `SeparationInteraction` moves the edge to a random voter who agrees with it.

    Args:
        label:
            The label of the voter or the group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        opinion:
            The opinion, 0 or 1.

        degree:
            The number of neighbours, read at the start of every step.

        activity:
            The probability that a voter acts in a step.

        rewiring:
            The probability that an active voter with a discordant neighbour rewires instead of copying.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import CoevolvingVoterElement
        >>> CoevolvingVoterElement('voters', rewiring=0.3).rewiring
        0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = OPINION_VISUAL,
        *,
        opinion: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        degree: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        activity: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        rewiring: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
    ) -> None:
        super().__init__(label, visual=visual)
        self.opinion = opinion
        self.degree = degree
        self.activity = activity
        self.rewiring = rewiring
