"""Spatial games: cooperation on a lattice where players imitate more successful neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element, LatticeElement
from ....core.interactions import GroupingInteraction, ImitationInteraction, LocalRuleInteraction
from ....core.laws.utils import count, total
from ....core.spaces import DiscreteTime, LatticeNeighborhood, LatticeSpace
from ....core.system import System
from ....errors import ValidationError
from ....specs import ParameterSpec, PhaseSpec, PhasesScheduleSpec, RoleSpec, StateSpec, VisualSpec
from ....types import Array, Pair
from ..._catalog import register_model

IMITATIONS = ('best', 'fermi')
ORIGIN: Pair = (0, 0)
MOORE: tuple[Pair, ...] = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)
SQUARE = LatticeNeighborhood(tuple((dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)))


def _shift(first: Pair, second: Pair) -> Pair:
    """Return the sum of two offsets."""
    return first[0] + second[0], first[1] + second[1]


def _imitate_fermi(
    own: Array,
    own_payoff: Array,
    payoffs: list[Array],
    strategies: list[Array],
    *,
    noise: float | Array,
    random: Array,
) -> Array:
    """Return strategies copied from a random neighbour with probability `1 / (1 + exp((P_own - P_other) / noise))`."""
    count = len(payoffs)
    chosen = np.minimum((random * count).astype(int), count - 1)
    uniform = random * count - chosen
    picked = np.take_along_axis(np.stack(payoffs), chosen[None], axis=0)[0]
    strategy = np.take_along_axis(np.stack(strategies), chosen[None], axis=0)[0]
    exponent = np.clip((own_payoff - picked) / np.maximum(noise, 1e-12), -50, 50)
    return np.where(uniform < 1 / (1 + np.exp(exponent)), strategy, own)


def _imitate_best(own: Array, own_payoff: Array, payoffs: list[Array], strategies: list[Array]) -> Array:
    """Return the strategy of the player with the largest payoff among a player and its neighbours, itself on ties."""
    best, following = own_payoff.copy(), own.copy()
    for payoff, strategy in zip(payoffs, strategies, strict=True):
        following = np.where(payoff > best, strategy, following)
        best = np.maximum(best, payoff)
    return following


def _random_strategies(
    shape: Sequence[int],
    share: float,
    random_state: int | np.random.Generator | None,
) -> Array:
    """Return strategies that are 1, cooperation, with a probability and 0, defection, otherwise."""
    return (np.random.default_rng(random_state).random(tuple(shape)) < share).astype(float)


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


class ContributorElement(Element):
    """Player of the spatial public goods game, a cooperator at 1 and a defector at 0, with its payoff.

    The play is a `PublicGoodsInteraction` of the lattice or a `HyperedgeGoodsInteraction` of a hypergraph, and the
    imitation an `ImitationInteraction` that reads the payoffs of the same step, which `public_goods_game` and
    `hypergraph_public_goods` add.

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
        >>> from skcomplex.models.social.cooperation import ContributorElement
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


class PrisonersDilemmaInteraction(LocalRuleInteraction):
    """Play of the weak prisoner's dilemma between neighbours, and imitation of more successful neighbours.

    It is a stochastic `LocalRuleInteraction` of the strategies over the 5 by 5 square around a player, a cooperator
    at 1 and a defector at 0. Every step each player plays the weak dilemma, with reward 1, temptation `temptation`,
    and 0 otherwise, against itself and its eight neighbours. It then copies the best player among them, or a random
    neighbour with the Fermi probability `1 / (1 + exp((P_own - P_other) / noise))`. The temptation and the noise are
    parameters of the players.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the players on the lattice.

        imitation:
            `'best'` for the best neighbour, or `'fermi'` for the Fermi rule.

    Raises:
        ValidationError:
            The imitation is unknown, raised by `check`.

    Examples:
        >>> from skcomplex.models.social.cooperation import PrisonersDilemmaInteraction
        >>> PrisonersDilemmaInteraction('play', target='players', imitation='fermi').check().imitation
        'fermi'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], imitation: str = 'best') -> None:
        super().__init__(label, target=target, state='strategy', rule=self._play, neighborhood=SQUARE, stochastic=True)
        self.imitation = imitation

    def _check(self) -> None:
        """Raise an error for an unknown imitation.

        Raises:
            ValidationError:
                The imitation is unknown.
        """
        super()._check()
        if self.imitation not in IMITATIONS:
            error_msg = f'Unknown imitation {self.imitation!r}. Use one of {", ".join(IMITATIONS)}.'
            raise ValidationError(error_msg)

    def _play(self, *values: Array, temptation: float | Array, noise: float | Array, random: Array) -> Array:
        """Return the strategies after the play of every player against its neighbours and the imitation."""
        cells = dict(zip(SQUARE.offsets, values, strict=True))

        def payoff(center: Pair) -> Array:
            own = cells[center]
            around = sum(cells[_shift(center, offset)] for offset in (*MOORE, ORIGIN))
            return own * around + (1 - own) * around * temptation

        payoffs = [payoff(offset) for offset in MOORE]
        strategies = [cells[offset] for offset in MOORE]
        if self.imitation == 'best':
            return _imitate_best(cells[ORIGIN], payoff(ORIGIN), payoffs, strategies)
        return _imitate_fermi(cells[ORIGIN], payoff(ORIGIN), payoffs, strategies, noise=noise, random=random)


class PublicGoodsInteraction(GroupingInteraction):
    """Play of the public goods game in the groups of a player and its four neighbours, which sets the payoffs.

    It is a symbolic `GroupingInteraction` of the von Neumann neighbourhoods of a lattice, a cooperator at 1 and a
    defector at 0. Every player joins five groups, the one centred on itself and those centred on its four neighbours.
    In each group a cooperator pays 1, and the pot multiplied by `synergy` is shared by the members, so the payoff of a
    player is the sum of its shares minus its contributions. The synergy is a parameter of the players.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the players on the lattice.

        phase:
            The phase of the step the play runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.models.social.cooperation import PublicGoodsInteraction
        >>> PublicGoodsInteraction('play', target='players').check().write
        'payoff'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], phase: str | None = None) -> None:
        super().__init__(
            label,
            target=target,
            groups='neighborhood',
            form='symbolic',
            write='payoff',
            function=self._get_share,
            phase=phase,
        )

    def _get_share(self, group: SimpleNamespace) -> Expression:
        """Return the share of the pot of a group for its member, minus the contribution of the member."""
        pot = group.member.synergy * total(group.members.strategy, over=group.members)
        return pot / count(over=group.members) - group.member.strategy


@register_model(
    field='social/cooperation',
    tags=('lattice', 'games', 'cooperation', 'evolution'),
    references=('Nowak and May 1992', 'Szabó and Fáth 2007', 'Argun et al. 2021', 'Gros 2015'),
    defaults={'until': 50, 'dt': 1},
)
def spatial_prisoners_dilemma(
    shape: Sequence[int] = (50, 50),
    *,
    temptation: float = 1.85,
    imitation: str = 'best',
    noise: float = 0.1,
    share: float = 0.9,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the spatial prisoner's dilemma of Nowak and May, with best-neighbour or Fermi imitation.

    Args:
        shape:
            The number of rows and columns of the periodic lattice.

        temptation:
            The payoff of a defector against a cooperator.

        imitation:
            `'best'` to copy the best neighbour, or `'fermi'` to copy a random neighbour with the Fermi probability.

        noise:
            The noise of the Fermi imitation.

        share:
            The share of cooperators at the start.

        random_state:
            The seed or generator of the initial strategies.

    Returns:
        system:
            The model, with the strategies `players.strategy`.

    Examples:
        >>> from skcomplex.models.social.cooperation import spatial_prisoners_dilemma
        >>> spatial_prisoners_dilemma((10, 10)).elements[0].check().prepare().size_
        100
    """
    players = LatticeElement(
        'players',
        element=PrisonerElement,
        space=LatticeSpace(tuple(shape)),
        values={
            'strategy': _random_strategies(shape, share, random_state),
            'temptation': temptation,
            'noise': noise,
        },
    )
    return System(
        elements=[players],
        interactions=[PrisonersDilemmaInteraction('play', target='players', imitation=imitation)],
    )


@register_model(
    field='social/cooperation',
    tags=('lattice', 'games', 'cooperation', 'public goods'),
    references=('Szabó and Hauert 2002', 'Szabó and Fáth 2007'),
    defaults={'until': 50, 'dt': 1},
)
def public_goods_game(
    shape: Sequence[int] = (50, 50),
    *,
    synergy: float = 4.5,
    noise: float = 0.5,
    share: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the spatial public goods game in groups of five neighbours, with Fermi imitation.

    Every step has two phases. In the phase `play` every player takes the payoff of its five groups, and in the phase
    `imitate` every player copies a random neighbour with the Fermi probability `1 / (1 + exp((P_own - P_other) / K))`
    of the payoffs of the same step, at the noise `K`.

    Args:
        shape:
            The number of rows and columns of the periodic lattice.

        synergy:
            The factor that multiplies the contributions of a group.

        noise:
            The noise of the Fermi imitation.

        share:
            The share of cooperators at the start.

        random_state:
            The seed or generator of the initial strategies.

    Returns:
        system:
            The model, with the strategies `players.strategy` and the payoffs `players.payoff`.

    Examples:
        >>> from skcomplex.models.social.cooperation import public_goods_game
        >>> public_goods_game((10, 10)).elements[0].check().prepare().size_
        100
    """
    strategies = _random_strategies(shape, share, random_state)
    players = LatticeElement(
        'players',
        element=ContributorElement,
        space=LatticeSpace(tuple(shape), neighborhood='von_neumann'),
        values={'strategy': strategies, 'synergy': synergy, 'noise': noise},
    )
    imitation = ImitationInteraction(
        'imitation',
        target='players',
        state='strategy',
        rule='fermi',
        form='probability',
        intensity=1 / noise,
        phase='imitate',
    )
    time = DiscreteTime(schedule=PhasesScheduleSpec((PhaseSpec('play'), PhaseSpec('imitate'))))
    return System(
        elements=[players],
        interactions=[PublicGoodsInteraction('play', target='players', phase='play'), imitation],
        time=time,
    )
