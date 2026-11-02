"""Player of the social dilemmas on a simplicial complex, a cooperator at 1 and a defector at 0, with its payoff."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class DilemmaElement(Element):
    """Player of the social dilemmas on a simplicial complex, a cooperator at 1 and a defector at 0, with its payoff.

    The play is a `SimplicialDilemmaInteraction` of the network, and the imitation an `ImitationInteraction` that
    reads the payoffs of the same step, which `simplicial_social_dilemmas` adds.

    Args:
        label:
            The label of the player or the group.

        visual:
            How the player is drawn, or `None` for `default_visual_`.

        strategy:
            The initial strategy, 1 or 0.

        payoff:
            The payoff of the last play, the mean payoff of the player over the links of its triangles.

        triangles:
            The number of triangles of the network that hold the player.

    Examples:
        >>> from skcomplex.models.social.cooperation import DilemmaElement
        >>> DilemmaElement('player').triangles
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        strategy: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 1.0,
        payoff: Annotated[float, StateSpec()] = 0.0,
        triangles: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.strategy = strategy
        self.payoff = payoff
        self.triangles = triangles
