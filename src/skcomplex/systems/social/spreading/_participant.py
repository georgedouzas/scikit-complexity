"""Participant in the environments of a contagion, susceptible at 0 and infected at 1."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ParticipantElement(Element):
    """Participant in the environments of a contagion, susceptible at 0 and infected at 1.

    The infection and the recovery are an `ExposureInteraction` of the hypergraph, which `hypergraph_exposure_sis`
    adds.

    Args:
        label:
            The label of the participant or the group.

        visual:
            How the participant is drawn, or `None` for `default_visual_`.

        infected:
            The initial state, 0 or 1.

        recovery:
            The probability that an infected participant recovers in a step.

    Examples:
        >>> from skcomplex.systems.social.spreading import ParticipantElement
        >>> ParticipantElement('participant').recovery
        0.05
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        infected: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        recovery: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.infected = infected
        self.recovery = recovery
