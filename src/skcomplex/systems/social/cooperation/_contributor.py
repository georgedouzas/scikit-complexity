"""Player of the spatial public goods game, a cooperator at 1 and a defector at 0, with its payoff."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ContributorElement(Element):
    """Player of the spatial public goods game, a cooperator at 1 and a defector at 0, with its payoff.

    The play is a `PublicGoodsInteraction` of the lattice or a `HyperedgeGoodsInteraction` of a hypergraph, and the
    imitation an `ImitationInteraction` that reads the payoffs of the same step, which `PublicGoodsGameSystem` and
    `HypergraphPublicGoodsSystem` add.

    Args:
        label:
            The label of the player or the lattice group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        strategy:
            The initial strategy, 1 or 0.

        payoff:
            The payoff of the last play, the sum of the shares of its groups minus its contributions.

        synergy:
            The factor that multiplies the contributions of a group.

        noise:
            The noise of the Fermi imitation.

    Examples:
        >>> from skcomplex.systems.social.cooperation import ContributorElement
        >>> ContributorElement('player').synergy
        4.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        strategy: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 1.0,
        payoff: Annotated[float, StateSpec()] = 0.0,
        synergy: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 4.5,
        noise: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.strategy = strategy
        self.payoff = payoff
        self.synergy = synergy
        self.noise = noise
