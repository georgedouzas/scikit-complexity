"""Player of the spatial prisoner's dilemma, a cooperator at 1 and a defector at 0."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class PrisonerElement(Element):
    """Player of the spatial prisoner's dilemma, a cooperator at 1 and a defector at 0.

    The play and the imitation are a `PrisonersDilemmaInteraction` of the lattice, which `spatial_prisoners_dilemma`
    adds.

    Args:
        label:
            The label of the player or the lattice group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        strategy:
            The initial strategy, 1 or 0.

        temptation:
            The payoff of a defector against a cooperator.

        noise:
            The noise of the Fermi imitation.

    Examples:
        >>> from skcomplex.models.social.cooperation import PrisonerElement
        >>> PrisonerElement('player').temptation
        1.85
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        strategy: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 1.0,
        temptation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.85,
        noise: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.strategy = strategy
        self.temptation = temptation
        self.noise = noise
