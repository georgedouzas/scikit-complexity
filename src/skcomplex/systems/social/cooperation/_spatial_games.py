"""Spatial games: cooperation on a lattice.

Players imitate more successful neighbours.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, Expression, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import GroupingInteraction, LocalRuleInteraction
from ....core.components.laws.utils import count, total
from ....core.components.spaces import BaseSpace, DiscreteTime, LatticeNeighborhood, LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, PhaseSpec, PhasesScheduleSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array, Pair
from ...general.agents import ImitationInteraction
from ._contributor import ContributorElement


class IteratedPrisonerElement(Element):
    """Player of the lattice iterated prisoner's dilemma.

    It cooperates for a number of rounds and then defects. The strategy `n` of a player is the number of rounds it
    intends to cooperate, until round `n` included or until its opponent defects, after which it defects for the rest of
    the game. The payoffs are years in prison, so lower is better. The play, the revision, and the mutation are an
    `IteratedDilemmaInteraction` of the lattice, which `IteratedLatticeDilemmaSystem` adds.

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


class PrisonerElement(Element):
    """Player of the spatial prisoner's dilemma, a cooperator at 1 and a defector at 0.

    The play and the imitation are a `PrisonersDilemmaInteraction` of the lattice, which `SpatialPrisonersDilemmaSystem`
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
        >>> from skcomplex.systems.social.cooperation import PrisonerElement
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


ORIGIN: Pair = (0, 0)


REACH = 2


HALF = 0.5


VON_NEUMANN: tuple[Pair, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


DIAMOND = LatticeNeighborhood(
    tuple((dx, dy) for dx in range(-REACH, REACH + 1) for dy in range(-REACH, REACH + 1) if abs(dx) + abs(dy) <= REACH),
)


class IteratedDilemmaInteraction(LocalRuleInteraction):
    """Play of the iterated prisoner's dilemma between neighbours.

    The play is followed by revision to the best strategy, and mutation. It is a stochastic `LocalRuleInteraction` of
    the strategies over the cells within two von Neumann steps of a player. Every step each player plays the `N`-round
    prisoner's dilemma with its four von Neumann neighbours and sums its years in prison. A player of strategy `n`
    against one of strategy `m < n` gets `m R + S + (N - m - 1) P`, the other `m R + T + (N - m - 1) P`, and two players
    of strategy `n` get `n R + (N - n) P`. Every player then adopts the strategy of the lowest score among itself and
    its four neighbours, a random one of them on ties, and finally mutates to a uniformly drawn strategy with the
    probability `mu`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the players on the lattice.

    Examples:
        >>> from skcomplex.systems.social.cooperation import IteratedDilemmaInteraction
        >>> IteratedDilemmaInteraction('play', target='players').state
        'strategy'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=target, state='strategy', rule=self._play, neighborhood=DIAMOND, stochastic=True)

    @staticmethod
    def years(
        own: Array,
        other: Array,
        *,
        rounds: float | Array,
        reward: float | Array,
        sucker: float | Array,
        temptation: float | Array,
        punishment: float | Array,
    ) -> Array:
        """Return the years in prison of a player of strategy `n` against strategy `m`.

        The game has `N` rounds.

        Args:
            own:
                The strategies `n` of the players.

            other:
                The strategies `m` of their opponents.

            rounds:
                The number `N` of rounds.

            reward:
                The punishment `R` of mutual cooperation.

            sucker:
                The punishment `S` of a betrayed cooperator.

            temptation:
                The punishment `T` of a defector who betrays.

            punishment:
                The punishment `P` of mutual defection.

        Returns:
            years:
                The years in prison of every player.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.systems.social.cooperation import IteratedDilemmaInteraction
            >>> years = IteratedDilemmaInteraction.years(
            ...     np.arange(11.0), np.full(11, 6.0), rounds=10, reward=0.5, sucker=1.5, temptation=0.0, punishment=1.0
            ... )
            >>> int(np.argmin(years))
            5
        """
        first = np.minimum(own, other)
        betrayal = np.where(own < other, temptation, sucker)
        mixed = first * reward + betrayal + (rounds - first - 1) * punishment
        return np.where(own == other, own * reward + (rounds - own) * punishment, mixed)

    def _play(
        self,
        *values: Array,
        rounds: float | Array,
        reward: float | Array,
        sucker: float | Array,
        temptation: float | Array,
        punishment: float | Array,
        mutation: float | Array,
        extremes: float | Array,
        random: Array,
    ) -> Array:
        """Return the strategies after the play, the revision, and the mutation.

        The play is against the neighbours.
        """
        cells = dict(zip(DIAMOND.offsets, values, strict=True))
        payoffs = {'rounds': rounds, 'reward': reward, 'sucker': sucker, 'temptation': temptation}

        def score(center: Pair) -> Array:
            own = cells[center]
            return sum(
                (
                    IteratedDilemmaInteraction.years(
                        own,
                        cells[center[0] + dx, center[1] + dy],
                        punishment=punishment,
                        **payoffs,
                    )
                    for dx, dy in VON_NEUMANN
                ),
                start=np.zeros_like(own),
            )

        candidates = (ORIGIN, *VON_NEUMANN)
        scores = np.stack([score(offset) for offset in candidates])
        strategies = np.stack([cells[offset] for offset in candidates])
        tied = np.isclose(scores, scores.min(axis=0))
        mutation = np.broadcast_to(mutation, random.shape)
        choice = np.clip((random - mutation) / np.maximum(1 - mutation, 1e-12), 0.0, 1.0 - 1e-12)
        rank = (choice * tied.sum(axis=0)).astype(int)
        picked = np.argmax(np.cumsum(tied, axis=0) > rank, axis=0)
        revised = np.take_along_axis(strategies, picked[None], axis=0)[0]
        draw = random / np.maximum(mutation, 1e-12)
        mutant = np.where(
            np.broadcast_to(extremes, random.shape) > 0,
            np.where(draw < HALF, 0.0, rounds),
            np.minimum(np.floor(draw * (np.asarray(rounds) + 1)), rounds),
        )
        return np.where(random < mutation, mutant, revised)


IMITATIONS = ('best', 'fermi')


MOORE: tuple[Pair, ...] = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)


SQUARE = LatticeNeighborhood(tuple((dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)))


class PrisonersDilemmaInteraction(LocalRuleInteraction):
    """Play of the weak prisoner's dilemma between neighbours.

    The play is followed by imitation of more successful neighbours. It is a stochastic `LocalRuleInteraction` of the
    strategies over the 5 by 5 square around a player, a cooperator at 1 and a defector at 0. Every step each player
    plays the weak dilemma, with reward 1, temptation `temptation`, and 0 otherwise, against itself and its eight
    neighbours. It then copies the best player among them, or a random neighbour with the Fermi probability
    `1 / (1 + exp((P_own - P_other) / noise))`. The temptation and the noise are parameters of the players.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the players on the lattice.

        imitation:
            `'best'` for the best neighbour, or `'fermi'` for the Fermi rule.

    Raises:
        ValidationError:
            The imitation is unknown, raised by `prepare`.

    Examples:
        >>> from skcomplex.systems.social.cooperation import PrisonersDilemmaInteraction
        >>> PrisonersDilemmaInteraction('play', target='players', imitation='fermi').imitation
        'fermi'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], imitation: str = 'best') -> None:
        super().__init__(label, target=target, state='strategy', rule=self._play, neighborhood=SQUARE, stochastic=True)
        self.imitation = imitation

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for an unknown imitation.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The imitation is unknown.
        """
        super()._prepare(context)
        if self.imitation not in IMITATIONS:
            error_msg = f'Unknown imitation {self.imitation!r}. Use one of {", ".join(IMITATIONS)}.'
            raise ValidationError(error_msg)

    def _play(self, *values: Array, temptation: float | Array, noise: float | Array, random: Array) -> Array:
        """Return the strategies after the play and the imitation.

        Every player plays against its neighbours.
        """
        cells = dict(zip(SQUARE.offsets, values, strict=True))

        def payoff(center: Pair) -> Array:
            own = cells[center]
            around = sum(cells[PrisonersDilemmaInteraction._shift(center, offset)] for offset in (*MOORE, ORIGIN))
            return own * around + (1 - own) * around * temptation

        payoffs = [payoff(offset) for offset in MOORE]
        strategies = [cells[offset] for offset in MOORE]
        if self.imitation == 'best':
            return PrisonersDilemmaInteraction._imitate_best(cells[ORIGIN], payoff(ORIGIN), payoffs, strategies)
        return PrisonersDilemmaInteraction._imitate_fermi(
            cells[ORIGIN],
            payoff(ORIGIN),
            payoffs,
            strategies,
            noise=noise,
            random=random,
        )

    @staticmethod
    def _shift(first: Pair, second: Pair) -> Pair:
        """Return the sum of two offsets."""
        return first[0] + second[0], first[1] + second[1]

    @staticmethod
    def _imitate_fermi(
        own: Array,
        own_payoff: Array,
        payoffs: list[Array],
        strategies: list[Array],
        *,
        noise: float | Array,
        random: Array,
    ) -> Array:
        """Return the strategy of a random neighbour with the Fermi probability.

        The probability is `1 / (1 + exp((P_own - P_other) / noise))`.
        """
        count = len(payoffs)
        chosen = np.minimum((random * count).astype(int), count - 1)
        uniform = random * count - chosen
        picked = np.take_along_axis(np.stack(payoffs), chosen[None], axis=0)[0]
        strategy = np.take_along_axis(np.stack(strategies), chosen[None], axis=0)[0]
        exponent = np.clip((own_payoff - picked) / np.maximum(noise, 1e-12), -50, 50)
        return np.where(uniform < 1 / (1 + np.exp(exponent)), strategy, own)

    @staticmethod
    def _imitate_best(own: Array, own_payoff: Array, payoffs: list[Array], strategies: list[Array]) -> Array:
        """Return the strategy of the largest payoff among a player and its neighbours.

        Ties keep its own.
        """
        best, following = own_payoff.copy(), own.copy()
        for payoff, strategy in zip(payoffs, strategies, strict=True):
            following = np.where(payoff > best, strategy, following)
            best = np.maximum(best, payoff)
        return following


class PublicGoodsInteraction(GroupingInteraction):
    """Play of the public goods game in the groups of a player and its neighbours.

    The groups hold a player and its four neighbours, and the play sets the payoffs. It is a symbolic
    `GroupingInteraction` of the von Neumann neighbourhoods of a lattice, a cooperator at 1 and a defector at 0. Every
    player joins five groups, the one centred on itself and those centred on its four neighbours. In each group a
    cooperator pays 1, and the pot multiplied by `synergy` is shared by the members, so the payoff of a player is the
    sum of its shares minus its contributions. The synergy is a parameter of the players.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the players on the lattice.

        phase:
            The phase of the step the play runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.social.cooperation import PublicGoodsInteraction
        >>> PublicGoodsInteraction('play', target='players').write
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
        """Return the share of the pot of a group for its member.

        The contribution of the member is subtracted.
        """
        pot = group.member.synergy * total(group.members.strategy, over=group.members)
        return pot / count(over=group.members) - group.member.strategy


def _random_strategies(
    shape: Sequence[int],
    share: float,
    random_state: int | np.random.Generator | None,
) -> Array:
    """Return strategies that are 1, cooperation, with a probability.

    The other strategies are 0, defection.
    """
    return (np.random.default_rng(random_state).random(tuple(shape)) < share).astype(float)


class SpatialPrisonersDilemmaSystem(BaseSystem):
    """The spatial prisoner's dilemma of Nowak and May.

    The imitation is of the best neighbour or by the Fermi rule. The system has the strategies `players.strategy`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.cooperation import SpatialPrisonersDilemmaSystem
        >>> SpatialPrisonersDilemmaSystem((10, 10)).prepare().system_.elements[0].prepare().size_
        100
    """

    field = 'social/cooperation'
    topics = ('lattice', 'games', 'cooperation', 'evolution')
    references = ('Nowak and May 1992', 'Szabó and Fáth 2007', 'Argun et al. 2021', 'Gros 2015')
    until = 50
    dt = 1

    def __init__(
        self,
        shape: Sequence[int] = (50, 50),
        *,
        temptation: float = 1.85,
        imitation: str = 'best',
        noise: float = 0.1,
        share: float = 0.9,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.temptation = temptation
        self.imitation = imitation
        self.noise = noise
        self.share = share
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        players = GroupElement(
            'players',
            element=PrisonerElement,
            space=LatticeSpace(tuple(self.shape)),
            values={
                'strategy': _random_strategies(self.shape, self.share, self.random_state),
                'temptation': self.temptation,
                'noise': self.noise,
            },
            on='sites',
        )
        return CompositeSystem(
            self.label,
            elements=[players],
            interactions=[PrisonersDilemmaInteraction('play', target='players', imitation=self.imitation)],
        )


class PublicGoodsGameSystem(BaseSystem):
    """The spatial public goods game in groups of five neighbours.

    The imitation follows the Fermi rule. Every step has two phases. In the phase `play` every player takes the payoff
    of its five groups, and in the phase `imitate` every player copies a random neighbour with the Fermi probability
    `1 / (1 + exp((P_own - P_other) / K))` of the payoffs of the same step, at the noise `K`. The system has the
    strategies `players.strategy` and the payoffs `players.payoff`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.cooperation import PublicGoodsGameSystem
        >>> PublicGoodsGameSystem((10, 10)).prepare().system_.elements[0].prepare().size_
        100
    """

    field = 'social/cooperation'
    topics = ('lattice', 'games', 'cooperation', 'public goods')
    references = ('Szabó and Hauert 2002', 'Szabó and Fáth 2007')
    until = 50
    dt = 1

    def __init__(
        self,
        shape: Sequence[int] = (50, 50),
        *,
        synergy: float = 4.5,
        noise: float = 0.5,
        share: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.synergy = synergy
        self.noise = noise
        self.share = share
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        strategies = _random_strategies(self.shape, self.share, self.random_state)
        players = GroupElement(
            'players',
            element=ContributorElement,
            space=LatticeSpace(tuple(self.shape), metric='manhattan'),
            values={'strategy': strategies, 'synergy': self.synergy, 'noise': self.noise},
            on='sites',
        )
        imitation = ImitationInteraction(
            'imitation',
            target='players',
            state='strategy',
            rule='fermi',
            form='probability',
            intensity=1 / self.noise,
            phase='imitate',
        )
        time = DiscreteTime(schedule=PhasesScheduleSpec((PhaseSpec('play'), PhaseSpec('imitate'))))
        return CompositeSystem(
            self.label,
            elements=[players],
            interactions=[PublicGoodsInteraction('play', target='players', phase='play'), imitation],
            time=time,
        )


class IteratedLatticeDilemmaSystem(BaseSystem):
    """Argun's evolutionary iterated prisoner's dilemma on a lattice.

    Its strategies are of `n` cooperation rounds. The payoffs are years in prison, `T < R < P < S`, so lower is better.
    A strategy `n` cooperates until round `n` or until the opponent defects, and defects afterwards, so `n = 0` always
    defects and `n = N` always cooperates. Against a strategy `m` the best reply is `n = m - 1`. Every step each player
    plays the `N`-round game with its four von Neumann neighbours, adopts the strategy of the lowest total among itself
    and its neighbours, random on ties, and mutates to a random strategy with the probability `mu` (Argun et al. 2021
    sections 13.2 and 13.3). With every strategy allowed and `S = 1.5`, cooperative strategies prevail for `R` below
    about 0.5, all strategies coexist and cycle between about 0.5 and 0.75, and defecting strategies dominate above, so
    the default `R = 0.6` lies in the middle of the cycling regime, where a finite lattice is not absorbed by defection.
    With only the two extreme strategies, `extremes=True`, cooperation dominates at `R = 0.82`, coexists with defection
    at `R = 0.84`, and vanishes at `R = 0.86` (Exercise 13.3).

    The system has the strategies `players.strategy`.

    Args:
        shape:
            The number of rows and columns of the periodic lattice.

        rounds:
            The number `N` of rounds of a game.

        reward:
            The punishment `R` of mutual cooperation.

        sucker:
            The punishment `S` of a betrayed cooperator.

        temptation:
            The punishment `T` of a defector who betrays.

        punishment:
            The punishment `P` of mutual defection.

        mutation:
            The probability `mu` of a mutation to a random strategy per player and step.

        extremes:
            Whether only the strategies 0, always defect, and `N`, always cooperate, are allowed.

        random_state:
            The seed or generator of the initial strategies.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.cooperation import IteratedLatticeDilemmaSystem
        >>> IteratedLatticeDilemmaSystem((10, 10)).prepare().system_.elements[0].prepare().size_
        100
    """

    field = 'social/cooperation'
    topics = ('lattice', 'games', 'cooperation', 'iterated', 'mutation')
    references = ('Argun et al. 2021', 'Axelrod and Hamilton 1981')
    until = 100
    dt = 1

    def __init__(
        self,
        shape: Sequence[int] = (30, 30),
        *,
        rounds: int = 7,
        reward: float = 0.6,
        sucker: float = 1.5,
        temptation: float = 0.0,
        punishment: float = 1.0,
        mutation: float = 0.01,
        extremes: bool = False,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.rounds = rounds
        self.reward = reward
        self.sucker = sucker
        self.temptation = temptation
        self.punishment = punishment
        self.mutation = mutation
        self.extremes = extremes
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        if self.extremes:
            strategies = self.rounds * rng.integers(2, size=tuple(self.shape)).astype(float)
        else:
            strategies = rng.integers(self.rounds + 1, size=tuple(self.shape)).astype(float)
        players = GroupElement(
            'players',
            element=IteratedPrisonerElement,
            space=LatticeSpace(tuple(self.shape), metric='manhattan'),
            values={
                'strategy': strategies,
                'rounds': self.rounds,
                'reward': self.reward,
                'sucker': self.sucker,
                'temptation': self.temptation,
                'punishment': self.punishment,
                'mutation': self.mutation,
                'extremes': float(self.extremes),
            },
            on='sites',
        )
        return CompositeSystem(
            self.label,
            elements=[players],
            interactions=[IteratedDilemmaInteraction('play', target='players')],
        )
