"""Game transition systems: donation games on the edges of a graph that change with the actions played."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.spaces import NetworkSpace
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array, IntArray


class DonorView(AgentView):
    """View of a player of the game transitions system, with its states as attributes."""

    strategy: float
    payoff: float


UPDATES = ('death-birth', 'pairwise')
MUTUAL_COOPERATION = 2


class DonorElement(Element):
    """Player of donation games on the edges of a graph, a cooperator at 1 and a defector at 0.

    Args:
        label:
            The label of the player or the group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        strategy:
            The strategy, 1 to cooperate and 0 to defect.

        payoff:
            The payoff accumulated over the games with every neighbour in the last time step.

    Examples:
        >>> from skcomplex.systems.social.cooperation import DonorElement
        >>> DonorElement('player').strategy
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'strategy', 'color_range': (0.0, 1.0)}),
        *,
        strategy: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 1.0,
        payoff: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.strategy = strategy
        self.payoff = payoff

    @classmethod
    def _get_view(cls) -> type[DonorView]:
        """Return the class of the views of the agents, whose attributes are the states."""
        return DonorView


class GameTransitionsInteraction(Interaction):
    """Evolutionary dynamics with game transitions, where the game on every edge follows the actions played on it.

    Every edge of the graph carries game 1 or game 2, donation games where a cooperator pays `c` to give its partner
    `b1` or `b2`, and every edge starts with game 1. In every time step every player plays the game of every edge with
    its neighbour and accumulates the payoff `pi`, of fitness `1 - delta + delta pi`. Under death-birth updating a
    uniformly drawn player dies and its neighbours compete for the site in proportion to their fitness, and under
    pairwise comparison a uniformly drawn player copies a random neighbour with the probability
    `1 / (1 + exp(-delta (pi_j - pi_i)))`. The games then change: an edge goes to game 2 with the probability
    `after_cooperation` after mutual cooperation, with `after_defection` after one defection, and surely after mutual
    defection, and to game 1 otherwise. The newcomer of a site plays the games of its prior occupant. Every step of the
    system has one time step per player, and the share of edges with game 1 is reported as `game_one`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of players on the nodes of a network.

        first_benefit:
            The benefit `b1` of the more valuable game 1.

        second_benefit:
            The benefit `b2` of game 2.

        cost:
            The cost `c` of cooperating.

        selection:
            The intensity `delta` of selection.

        after_cooperation:
            The probability `p` of game 2 after mutual cooperation.

        after_defection:
            The probability `q` of game 2 after one cooperation and one defection.

        transitions:
            Whether the games change, or every edge plays game 1 throughout.

        update:
            `'death-birth'` or `'pairwise'`.

    Raises:
        ValidationError:
            The update is unknown, raised by `prepare`.

    Examples:
        >>> from skcomplex.systems.social.cooperation import GameTransitionsInteraction
        >>> GameTransitionsInteraction('evolution', target='players').update
        'death-birth'
    """

    draws = True

    law_classes = (RuleLaw,)

    reports: ClassVar[tuple[str, ...]] = ('game_one',)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        first_benefit: Annotated[float, ParameterSpec()] = 3.5,
        second_benefit: Annotated[float, ParameterSpec()] = 2.6,
        cost: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        selection: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        after_cooperation: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.0,
        after_defection: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
        transitions: bool = True,
        update: str = 'death-birth',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.first_benefit = first_benefit
        self.second_benefit = second_benefit
        self.cost = cost
        self.selection = selection
        self.after_cooperation = after_cooperation
        self.after_defection = after_defection
        self.transitions = transitions
        self.update = update

    def _check(self) -> None:
        """Raise an error for an unknown update.

        Raises:
            ValidationError:
                The update is unknown.
        """
        super()._check()
        if self.update not in UPDATES:
            error_msg = f'Unknown update {self.update!r}. Use one of {", ".join(UPDATES)}.'
            raise ValidationError(error_msg)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the evolution rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.evolve)]

    def _get_edges(self, players: list[DonorView]) -> tuple[IntArray, IntArray]:
        """Return the two ends of every edge, as indices of the players."""
        index = {player.id: position for position, player in enumerate(players)}
        pairs = sorted(
            {
                (min(index[player.id], index[other.id]), max(index[player.id], index[other.id]))
                for player in players
                for other in player.neighbors()
                if isinstance(other, DonorView)
            },
        )
        ends = np.array(pairs, dtype=int).reshape(-1, 2)
        return ends[:, 0], ends[:, 1]

    def _get_payoffs(self, strategies: Array, games: Array, first: IntArray, second: IntArray) -> Array:
        """Return the payoff of every player over the games of all its edges."""
        benefits = np.where(games == 1, self.first_benefit, self.second_benefit)
        gains = benefits * strategies[second] - self.cost * strategies[first]
        returns = benefits * strategies[first] - self.cost * strategies[second]
        return np.bincount(first, gains, strategies.size) + np.bincount(second, returns, strategies.size)

    def _replace(self, strategies: Array, payoffs: Array, neighbors: list[IntArray], rng: np.random.Generator) -> None:
        """Replace the strategy of one uniformly drawn player by death-birth or pairwise comparison."""
        player = int(rng.integers(strategies.size))
        around = neighbors[player]
        if around.size == 0:
            return
        if self.update == 'pairwise':
            other = int(around[rng.integers(around.size)])
            exponent = np.clip(-self.selection * (payoffs[other] - payoffs[player]), -500.0, 500.0)
            if rng.random() < 1 / (1 + np.exp(exponent)):
                strategies[player] = strategies[other]
            return
        fitness = np.maximum(1 - self.selection + self.selection * payoffs[around], 1e-12)
        strategies[player] = strategies[around[rng.choice(around.size, p=fitness / fitness.sum())]]

    def evolve(self, system: SystemView) -> None:
        """Run one time step per player: play the games, replace one player, and change the games.

        Args:
            system:
                The system.
        """
        players = system.agents(self.target, DonorView)
        storage = system.storage
        if 'first' not in storage:
            sources, targets = self._get_edges(players)
            storage['first'], storage['second'] = sources, targets
            storage['games'] = np.ones(sources.size)
            storage['neighbors'] = [
                np.concatenate([targets[sources == node], sources[targets == node]]) for node in range(len(players))
            ]
        first, second, games = storage['first'], storage['second'], storage['games']
        neighbors = storage['neighbors']
        if not isinstance(first, np.ndarray) or not isinstance(second, np.ndarray) or not isinstance(games, np.ndarray):
            return
        if not isinstance(neighbors, list):
            return
        rng = system.rng
        strategies = np.array([player.strategy for player in players])
        payoffs = np.zeros(len(players))
        for _ in range(len(players)):
            payoffs = self._get_payoffs(strategies, games, first, second)
            actions = strategies[first] + strategies[second]
            self._replace(strategies, payoffs, neighbors, rng)
            if self.transitions:
                second_game = np.where(actions == MUTUAL_COOPERATION, self.after_cooperation, self.after_defection)
                second_game = np.where(actions == 0, 1.0, second_game)
                games[:] = np.where(rng.random(games.size) < second_game, 2.0, 1.0)
        for player, strategy, payoff in zip(players, strategies.tolist(), payoffs.tolist(), strict=True):
            player.strategy = strategy
            player.payoff = payoff
        system.report('game_one', float(np.mean(games == 1)) if games.size else 1.0)


class GameTransitionsSystem(BaseSystem):
    """Evolutionary dynamics with game transitions on a random regular graph, after Su, McAvoy, Wang, and Nowak.

    Every edge carries a donation game, game 1 of benefit `b1` or game 2 of benefit `b2 < b1`, at the same cost `c`.
    Players accumulate the payoffs of the games with all their neighbours, one player is replaced every time step by
    death-birth updating or pairwise comparison, and the game of every edge changes with the actions played on it:
    with the default transitions mutual cooperation leads to game 1 and every other outcome to game 2 (arXiv 1905.10269
    section 2 and section 3.1). Under weak selection and death-birth updating on a `k`-regular graph, cooperation is
    favoured if `b1 / c > k - xi (b1 - b2) / c` with `xi = (k - 1) / 2`, lowered from the rule `b1 / c > k` of a
    single game, and with the probabilities `p` and `q` of game 2 after mutual cooperation and after one defection,
    `xi = (k - 1) q / 2 - (k + 1) p / 2` (Eq. 1 and Eq. 3). Under pairwise comparison cooperation is favoured if
    `(b1 - b2) / (2 c) > 1` (Eq. 2), whatever `b1`, and never with a single game. The defaults `b1 = 3.5`,
    `b2 = b1 - 0.9`, and `k = 4` lie in the range of Fig. 2 where transitions favour cooperation, since
    `3.5 > 4 - 1.5 * 0.9`, and game 1 alone does not, since `3.5 < 4`.

    The system has the group `players` on the nodes of the graph, and the reporter `evolution.game_one`.

    Args:
        n_players:
            The number of players.

        degree:
            The number `k` of neighbours of every player.

        first_benefit:
            The benefit `b1` of game 1.

        second_benefit:
            The benefit `b2` of game 2.

        cost:
            The cost `c` of cooperating.

        selection:
            The intensity `delta` of selection.

        after_cooperation:
            The probability `p` of game 2 after mutual cooperation.

        after_defection:
            The probability `q` of game 2 after one cooperation and one defection.

        transitions:
            Whether the games change, or every edge plays game 1 throughout.

        update:
            `'death-birth'` or `'pairwise'`.

        share:
            The share of cooperators at the start.

        random_state:
            The seed or generator of the graph and of the initial strategies.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.cooperation import GameTransitionsSystem
        >>> GameTransitionsSystem(20).prepare().system_.elements[0].prepare().size_
        20
    """

    field = 'social/cooperation'
    topics = ('network', 'games', 'cooperation', 'game transitions', 'evolution')
    references = ('Su, McAvoy, Wang, and Nowak 2019',)
    until = 100

    def __init__(
        self,
        n_players: int = 100,
        *,
        degree: int = 4,
        first_benefit: float = 3.5,
        second_benefit: float = 2.6,
        cost: float = 1.0,
        selection: float = 0.1,
        after_cooperation: float = 0.0,
        after_defection: float = 1.0,
        transitions: bool = True,
        update: str = 'death-birth',
        share: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_players = n_players
        self.degree = degree
        self.first_benefit = first_benefit
        self.second_benefit = second_benefit
        self.cost = cost
        self.selection = selection
        self.after_cooperation = after_cooperation
        self.after_defection = after_defection
        self.transitions = transitions
        self.update = update
        self.share = share
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        graph = nx.random_regular_graph(self.degree, self.n_players, seed=int(rng.integers(2**31)))
        players = GroupElement(
            'players',
            element=DonorElement,
            size=self.n_players,
            values={'strategy': (rng.random(self.n_players) < self.share).astype(float)},
        )
        evolution = GameTransitionsInteraction(
            'evolution',
            target='players',
            first_benefit=self.first_benefit,
            second_benefit=self.second_benefit,
            cost=self.cost,
            selection=self.selection,
            after_cooperation=self.after_cooperation,
            after_defection=self.after_defection,
            transitions=self.transitions,
            update=self.update,
        )
        return System(self.label, elements=[players], interactions=[evolution], space=NetworkSpace(graph))
