"""Player of the lattice iterated prisoner's dilemma, who cooperates for a number of rounds and then defects."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class IteratedPrisonerElement(Element):
    """Player of the lattice iterated prisoner's dilemma, who cooperates for a number of rounds and then defects.

    The strategy `n` of a player is the number of rounds it intends to cooperate, until round `n` included or until its
    opponent defects, after which it defects for the rest of the game. The payoffs are years in prison, so lower is
    better. The play, the revision, and the mutation are an `IteratedDilemmaInteraction` of the lattice, which
    `iterated_lattice_dilemma` adds.

    Args:
        label:
            The label of the player or the lattice group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        strategy:
            The number `n` of intended cooperation rounds, from 0, always defect, to `rounds`, always cooperate.

        rounds:
            The number `N` of rounds of a game.

        reward:
            The punishment `R` of mutual cooperation.

        sucker:
            The punishment `S` of a cooperator who is betrayed.

        temptation:
            The punishment `T` of a defector who betrays.

        punishment:
            The punishment `P` of mutual defection.

        mutation:
            The probability `mu` that a player mutates to a random strategy at the end of a step.

        extremes:
            1 to allow only the strategies `0` and `N`, and 0 to allow every strategy from 0 to `N`.

    Examples:
        >>> from skcomplex.systems.social.cooperation import IteratedPrisonerElement
        >>> IteratedPrisonerElement('player').rounds
        7
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        strategy: Annotated[float, StateSpec(domain=NN)] = 0.0,
        rounds: Annotated[int, ParameterSpec(domain=NN)] = 7,
        reward: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.72,
        sucker: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.5,
        temptation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        punishment: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        mutation: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.01,
        extremes: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.strategy = strategy
        self.rounds = rounds
        self.reward = reward
        self.sucker = sucker
        self.temptation = temptation
        self.punishment = punishment
        self.mutation = mutation
        self.extremes = extremes
