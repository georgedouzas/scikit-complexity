"""Hypergraph games: public goods in hyperedges, and dilemmas on a complex.

The complex is simplicial. The dilemmas are two- and three-player.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import xgi
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import CouplingInteraction
from ....core.components.laws.utils import count, total, where
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ParameterSpec, PhaseSpec, PhasesScheduleSpec, RoleSpec, StateSpec, VisualSpec
from ...general.agents import ImitationInteraction
from ._contributor import ContributorElement

PAIR = 2
CLIQUE = 5
PHASES = ('play', 'imitate')


class DilemmaElement(Element):
    """Player of the social dilemmas on a simplicial complex, with its payoff.

    It is a cooperator at 1 and a defector at 0. The play is a `SimplicialDilemmaInteraction` of the network, and the
    imitation an `ImitationInteraction` that reads the payoffs of the same step, which `SimplicialSocialDilemmasSystem`
    adds.

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
        >>> from skcomplex.systems.social.cooperation import DilemmaElement
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


class HyperedgeGoodsInteraction(CouplingInteraction):
    """Play of the public goods game in every hyperedge around a player.

    The play sets the payoffs. It is a `CouplingInteraction` over the hyperedges of a network in the map form, a
    cooperator at 1 and a defector at 0. In every hyperedge of `m` members a cooperator pays 1, and the pot multiplied
    by `synergy` is shared by the `m` members, so the payoff of a player is the sum over its hyperedges of its share
    minus its contribution. The synergy is a parameter of the players.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the players on the network.

        phase:
            The phase of the step the play runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.social.cooperation import HyperedgeGoodsInteraction
        >>> HyperedgeGoodsInteraction('play', target='players').write
        'payoff'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], phase: str | None = None) -> None:
        super().__init__(
            label,
            target=target,
            mode='hyperedges',
            form='map',
            write='payoff',
            function=self._get_share,
            phase=phase,
        )

    def _get_share(self, hyperedge: SimpleNamespace) -> Expression:
        """Return the share of the pot of a hyperedge for its member.

        The contribution of the member is subtracted.
        """
        pot = hyperedge.member.synergy * total(hyperedge.members.strategy, over=hyperedge.members)
        return pot / count(over=hyperedge.members) - hyperedge.member.strategy


class SimplicialDilemmaInteraction(CouplingInteraction):
    """Play of two-player games along the links of the triangles of a complex.

    The complex is simplicial. The play sets the payoffs. It is a `CouplingInteraction` over the hyperedges of a network
    in the map form, a cooperator at 1 and a defector at 0. Every game pays 1 for mutual cooperation, 0 for mutual
    defection, and the sucker's payoff `S` and the temptation `T` when the strategies differ. A triangle of three links
    is three hyperedges of two members, along which the players play game 1 of `S1` and `T1`. A triangle filled as a
    2-simplex is one hyperedge of three members, where a player plays each other member while the third looks on: game 2
    of `S2` and `T2` when the third member shares the player's strategy, and game 3 of `S3` and `T3` otherwise. The
    payoff of a player is the sum over the two links of every triangle around it divided by twice its number of
    triangles, a parameter of the players.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the players on the network.

        sucker:
            The payoff `S1` of a cooperator against a defector along a link.

        temptation:
            The payoff `T1` of a defector against a cooperator along a link.

        sucker_allied:
            The payoff `S2` in a 2-simplex whose third member shares the player's strategy.

        temptation_allied:
            The payoff `T2` in a 2-simplex whose third member shares the player's strategy.

        sucker_opposed:
            The payoff `S3` in a 2-simplex whose third member holds the other strategy.

        temptation_opposed:
            The payoff `T3` in a 2-simplex whose third member holds the other strategy.

        phase:
            The phase of the step the play runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.social.cooperation import SimplicialDilemmaInteraction
        >>> SimplicialDilemmaInteraction('play', target='players', temptation_allied=0.8).temptation_allied
        0.8
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        sucker: Annotated[float, ParameterSpec()] = -0.2,
        temptation: Annotated[float, ParameterSpec()] = 1.2,
        sucker_allied: Annotated[float, ParameterSpec()] = 0.5,
        temptation_allied: Annotated[float, ParameterSpec()] = 0.8,
        sucker_opposed: Annotated[float, ParameterSpec()] = -0.2,
        temptation_opposed: Annotated[float, ParameterSpec()] = 1.2,
        phase: str | None = None,
    ) -> None:
        super().__init__(
            label,
            target=target,
            mode='hyperedges',
            form='map',
            write='payoff',
            function=self._get_games,
            combine=self._get_mean,
            phase=phase,
        )
        self.sucker = sucker
        self.temptation = temptation
        self.sucker_allied = sucker_allied
        self.temptation_allied = temptation_allied
        self.sucker_opposed = sucker_opposed
        self.temptation_opposed = temptation_opposed

    def _get_games(self, hyperedge: SimpleNamespace) -> Expression:
        """Return the payoff of a member from the games along its links in one hyperedge.

        The hyperedge has two or three members. In a link a cooperator earns 1 or `S1` and a defector `T1` or 0 against
        a cooperator or a defector. In a 2-simplex with `c` cooperating others a cooperator earns 2, `1 + S2`, or `2
        S3`, and a defector `2 T3`, `T2`, or 0, for `c` of 2, 1, or 0.
        """
        own = hyperedge.member.strategy
        others = total(hyperedge.members.strategy, over=hyperedge.members) - own
        link = own * (others + (1 - others) * hyperedge.sucker) + (1 - own) * others * hyperedge.temptation
        cooperator = where(
            others == PAIR,
            PAIR,
            where(others == 1, 1 + hyperedge.sucker_allied, PAIR * hyperedge.sucker_opposed),
        )
        defector = where(
            others == PAIR,
            PAIR * hyperedge.temptation_opposed,
            where(others == 1, hyperedge.temptation_allied, 0),
        )
        simplex = own * cooperator + (1 - own) * defector
        return where(count(over=hyperedge.members) == PAIR, link, simplex)

    @staticmethod
    def _get_mean(player: SimpleNamespace, games: Expression) -> Expression:
        """Return the payoffs of the games of a player over its triangles.

        They are divided by twice its number of triangles.
        """
        return games / (PAIR * player.member.triangles)


class HypergraphPublicGoodsSystem(BaseSystem):
    """The public goods game on a hypergraph, played in every hyperedge.

    The imitation follows the Fermi rule. Every hyperedge is a group that plays the public goods game: a cooperator pays
    1, and the pot multiplied by `synergy` is shared by the members. Every step has two phases. In the phase `play`
    every player takes the sum of its shares minus its contributions over the hyperedges around it, and in the phase
    `imitate` every player copies a random neighbour, a member of one of its hyperedges, with the Fermi probability
    `1 / (1 + exp((P_own - P_other) / K))` of the payoffs of the same step, at the noise `K`. In a well-mixed population
    defection wins in a group of `m` for `synergy < m`, so cooperation needs a synergy near the sizes of the hyperedges.

    The system has the strategies `players.strategy` and the payoffs `players.payoff`.

    Args:
        n_nodes:
            The number of players.

        sizes:
            The sizes of the hyperedges.

        hyperdegree:
            The mean number of hyperedges of every size around a player, whose members are drawn at random.

        synergy:
            The factor that multiplies the contributions of a hyperedge.

        noise:
            The noise of the Fermi imitation.

        share:
            The share of cooperators at the start.

        seed:
            The seed of the hypergraph and the initial strategies.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.cooperation import HypergraphPublicGoodsSystem
        >>> HypergraphPublicGoodsSystem(30).prepare().system_.elements[0].prepare().size_
        30
    """

    field = 'social/cooperation'
    topics = ('hypergraph', 'games', 'cooperation', 'public goods')
    references = ('Alvarez-Rodriguez et al. 2021', 'Majhi et al. 2022')
    until = 100
    dt = 1

    def __init__(
        self,
        n_nodes: int = 400,
        *,
        sizes: Sequence[int] = (3, 5),
        hyperdegree: float = 2.0,
        synergy: float = 4.0,
        noise: float = 0.5,
        share: float = 0.5,
        seed: int = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_nodes = n_nodes
        self.sizes = sizes
        self.hyperdegree = hyperdegree
        self.synergy = synergy
        self.noise = noise
        self.share = share
        self.seed = seed

    @staticmethod
    def _get_hypergraph(
        n_nodes: int,
        sizes: Sequence[int],
        hyperdegree: float,
        rng: np.random.Generator,
    ) -> xgi.Hypergraph:
        """Return a hypergraph of hyperedges of random members.

        On average a node has `hyperdegree` hyperedges of every size around it.
        """
        hypergraph = xgi.Hypergraph()
        hypergraph.add_nodes_from(range(n_nodes))
        for size in sizes:
            number = round(n_nodes * hyperdegree / size)
            hypergraph.add_edges_from(rng.choice(n_nodes, size, replace=False).tolist() for _ in range(number))
        return hypergraph

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        rng = np.random.default_rng(self.seed)
        hypergraph = self._get_hypergraph(self.n_nodes, self.sizes, self.hyperdegree, rng)
        graph = xgi.to_graph(hypergraph)
        strategies = (rng.random(self.n_nodes) < self.share).astype(float)
        players = GroupElement(
            'players',
            element=ContributorElement,
            space=NetworkSpace(graph, hyperedges=hypergraph),
            values={'strategy': strategies, 'synergy': self.synergy, 'noise': self.noise},
            visual=VisualSpec('sites', {'states': ['strategy'], 'low': 0.0, 'high': 1.0}),
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
        return CompositeSystem(
            self.label,
            elements=[players],
            interactions=[HyperedgeGoodsInteraction('play', target='players', phase='play'), imitation],
            time=DiscreteTime(schedule=PhasesScheduleSpec(tuple(PhaseSpec(phase) for phase in PHASES))),
        )


class SimplicialSocialDilemmasSystem(BaseSystem):
    """The social dilemmas of Guo et al. on a simplicial complex.

    They are played on the links and the 2-simplices of the complex. The network grows from a complete graph on five
    nodes, every new node joining both ends of a random link, which forms one new triangle. A share `simplices` of the
    triangles, drawn at random, are 2-simplices, and the others are three links. Players play game 1 along the links of
    a plain triangle, and in a 2-simplex game 2 against a member when the third member shares their strategy and game 3
    otherwise, every game paying 1 for mutual cooperation, 0 for mutual defection, and `S` and `T` to a cooperator and a
    defector that meet. The payoff of a player is the mean over the two links of every triangle around it. Every step
    has two phases: in `play` every player takes its payoff, and in `imitate` every player copies a random neighbour
    with the Fermi probability `1 / (1 + exp((P_own - P_other) / K))` at the noise `K`.

    The review of Majhi et al. writes the payoff as a mean over the links of a player of the mean over the triangles of
    each link, and picks game 2 when `s_ki = s_kj`. This system takes the mean over the links of the triangles, which
    counts a link once per triangle, and reads the condition as the third member sharing the strategy of the player. The
    system has the strategies `players.strategy` and the payoffs `players.payoff`.

    Args:
        n_nodes:
            The number of players.

        simplices:
            The share of the triangles that are 2-simplices.

        sucker:
            The sucker's payoff `S1` of game 1.

        temptation:
            The temptation `T1` of game 1.

        sucker_allied:
            The sucker's payoff `S2` of game 2.

        temptation_allied:
            The temptation `T2` of game 2.

        sucker_opposed:
            The sucker's payoff `S3` of game 3, or `None` for `S1`.

        temptation_opposed:
            The temptation `T3` of game 3, or `None` for `T1`.

        noise:
            The noise of the Fermi imitation.

        share:
            The share of cooperators at the start.

        seed:
            The seed of the network, the 2-simplices, and the initial strategies.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.cooperation import SimplicialSocialDilemmasSystem
        >>> SimplicialSocialDilemmasSystem(30).prepare().system_.elements[0].prepare().size_
        30
    """

    field = 'social/cooperation'
    topics = ('simplicial complex', 'games', 'cooperation', 'social dilemmas')
    references = ('Guo et al. 2021', 'Majhi et al. 2022')
    until = 100
    dt = 1

    def __init__(
        self,
        n_nodes: int = 500,
        *,
        simplices: float = 0.5,
        sucker: float = -0.2,
        temptation: float = 1.2,
        sucker_allied: float = 0.5,
        temptation_allied: float = 0.8,
        sucker_opposed: float | None = None,
        temptation_opposed: float | None = None,
        noise: float = 0.1,
        share: float = 0.5,
        seed: int = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_nodes = n_nodes
        self.simplices = simplices
        self.sucker = sucker
        self.temptation = temptation
        self.sucker_allied = sucker_allied
        self.temptation_allied = temptation_allied
        self.sucker_opposed = sucker_opposed
        self.temptation_opposed = temptation_opposed
        self.noise = noise
        self.share = share
        self.seed = seed

    @staticmethod
    def _get_triangle_attachment(n_nodes: int, rng: np.random.Generator) -> tuple[nx.Graph, list[tuple[int, ...]]]:
        """Return a network grown from a complete graph on five nodes.

        Each new node joins both ends of a link.
        """
        graph = nx.complete_graph(CLIQUE)
        triangles: list[tuple[int, ...]] = list(itertools.combinations(range(CLIQUE), 3))
        edges = list(graph.edges)
        for node in range(CLIQUE, n_nodes):
            first, second = edges[rng.integers(len(edges))]
            graph.add_edges_from([(node, first), (node, second)])
            edges.extend([(node, first), (node, second)])
            triangles.append((min(first, second), max(first, second), node))
        return graph, triangles

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        rng = np.random.default_rng(self.seed)
        graph, triangles = self._get_triangle_attachment(self.n_nodes, rng)
        filled = rng.random(len(triangles)) < self.simplices
        hyperedges: list[tuple[int, ...]] = []
        for triangle, simplex in zip(triangles, filled, strict=True):
            hyperedges.extend([triangle] if simplex else itertools.combinations(triangle, PAIR))
        held = np.bincount(np.asarray(triangles).ravel(), minlength=self.n_nodes).astype(float)
        players = GroupElement(
            'players',
            element=DilemmaElement,
            space=NetworkSpace(graph, hyperedges=hyperedges),
            values={'strategy': (rng.random(self.n_nodes) < self.share).astype(float), 'triangles': held},
            visual=VisualSpec('sites', {'states': ['strategy'], 'low': 0.0, 'high': 1.0}),
            on='sites',
        )
        play = SimplicialDilemmaInteraction(
            'play',
            target='players',
            sucker=self.sucker,
            temptation=self.temptation,
            sucker_allied=self.sucker_allied,
            temptation_allied=self.temptation_allied,
            sucker_opposed=self.sucker if self.sucker_opposed is None else self.sucker_opposed,
            temptation_opposed=self.temptation if self.temptation_opposed is None else self.temptation_opposed,
            phase='play',
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
        time = DiscreteTime(schedule=PhasesScheduleSpec(tuple(PhaseSpec(phase) for phase in PHASES)))
        return CompositeSystem(self.label, elements=[players], interactions=[play, imitation], time=time)
