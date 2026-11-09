"""Evolution systems: evolutionary games.

The strategies spread with their payoff.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import LocalRuleInteraction
from ....core.components.laws.utils import total
from ....core.components.spaces import BallNeighborhood, LatticeSpace
from ....core.components.systems import BaseSystem, System
from ....core.specs import StateSpec, VisualSpec
from ....core.types import Array
from ..common.populations import ReplicatorInteraction

NEAREST = BallNeighborhood(metric='manhattan')

STRATEGIES = 3


class CyclicSpeciesElement(Element):
    """Site of a lattice of rock, paper, and scissors, 0, 1, and 2.

    Each strategy beats the one before it.

    The sites update one at a time in random order by a `LocalRuleInteraction`, and a site adopts the strategy of a
    random neighbour that beats it.

    Args:
        label:
            The label of the site or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        strategy:
            The initial strategy, 0, 1, or 2.

    Examples:
        >>> from skcomplex.systems.biology.evolution import CyclicSpeciesElement
        >>> CyclicSpeciesElement('site', strategy=2).strategy
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        strategy: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.strategy = strategy


class StrategyElement(Element):
    """Strategy of an evolutionary game, with the share `x` of players.

    The share is the one of the population that plays the strategy.

    Args:
        label:
            The label of the strategy or of the group of strategies.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial share.

    Examples:
        >>> from skcomplex.systems.biology.evolution import StrategyElement
        >>> StrategyElement('hawks', x=0.4).x
        0.4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x


class ReplicatorDynamicsSystem(BaseSystem):
    """The replicator dynamics of a game, by default hawks and doves.

    Hawks and doves settle at a share `V / C` of hawks.

    Every strategy is a member of the group `population`, and it grows with its payoff above the mean payoff. The hawks
    and doves game has the payoff matrix `((V - C) / 2, V), (0, V / 2)` with hawks first. When the cost of a fight `C`
    exceeds the value `V`, hawks and doves coexist with `V / C` hawks, and otherwise hawks take over. A payoff matrix
    gives any other game, such as the prisoner's dilemma or rock, paper, scissors. The system has the shares
    `population.x` of the strategies, hawks first.

    Args:
        x:
            The initial shares of the strategies, or `None` for equal shares.

        payoff:
            The payoff matrix of another game, one row per strategy, or `None` for hawks and doves.

        V:
            The value of the resource contested by hawks and doves.

        C:
            The cost of a fight between hawks.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.evolution import ReplicatorDynamicsSystem
        >>> system = ReplicatorDynamicsSystem(payoff=((0, -1, 1), (1, 0, -1), (-1, 1, 0))).prepare().system_
        >>> system.elements[0].prepare().size_
        3
    """

    field = 'biology/evolution'
    topics = ('evolutionary game', 'replicator', 'ode', 'game theory')
    references = ('Taylor and Jonker 1978', 'Maynard Smith 1982', 'Gros 2015', 'Szabó and Fáth 2007')
    until = 30.0
    dt = 0.1

    def __init__(
        self,
        x: Sequence[float] | None = None,
        *,
        payoff: Sequence[Sequence[float]] | None = None,
        V: float = 2.0,
        C: float = 3.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.payoff = payoff
        self.V = V
        self.C = C

    def _build_system(self) -> System:
        """Return the parts of the replicator dynamics of a game."""
        matrix = (
            (((self.V - self.C) / 2, self.V), (0.0, self.V / 2))
            if self.payoff is None
            else tuple(tuple(row) for row in self.payoff)
        )
        strategies = len(matrix)
        shares = [1 / strategies] * strategies if self.x is None else list(self.x)
        population = GroupElement('population', element=StrategyElement, size=strategies, values={'x': shares})
        weights = np.asarray(matrix, dtype=float)
        replication = ReplicatorInteraction(
            'replication',
            target='population',
            fitness=lambda s: total(s.target.x, weights=weights),
        )
        return System(self.label, elements=[population], interactions=[replication])


class RockPaperScissorsSystem(BaseSystem):
    """Rock, paper, and scissors on a lattice.

    Cyclic invasions keep the three strategies coexisting.

    Starting from random strategies, the strategies chase each other in domains and fronts, and each keeps about a third
    of the lattice. The system has the strategies `lattice.strategy`.

    Args:
        shape:
            The number of rows and columns of the periodic lattice.

        random_state:
            The seed or generator of the initial strategies.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.evolution import RockPaperScissorsSystem
        >>> RockPaperScissorsSystem((10, 10)).prepare().system_.elements[0].prepare().size_
        100
    """

    field = 'biology/evolution'
    topics = ('evolutionary game', 'lattice', 'cyclic dominance', 'spirals', 'stochastic')
    references = ('Szabó and Fáth 2007', 'Reichenbach, Mobilia, and Frey 2007')
    until = 100
    dt = 1

    def __init__(
        self,
        shape: Sequence[int] = (60, 60),
        *,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.random_state = random_state

    @staticmethod
    def _invade(strategy: Array, *around: Array, random: Array) -> Array:
        """Return the strategy of every site after invasion.

        A random neighbour that beats a site invades it.
        """
        choice = np.minimum((random * len(around)).astype(int), len(around) - 1)
        neighbor = np.choose(choice, np.stack(around))
        beaten = neighbor == np.mod(strategy + 1, STRATEGIES)
        return np.where(beaten, neighbor, strategy)

    def _build_system(self) -> System:
        """Return the parts of rock, paper, and scissors on a lattice."""
        strategies = (
            np.random.default_rng(self.random_state).integers(0, STRATEGIES, size=tuple(self.shape)).astype(float)
        )
        return System(
            self.label,
            elements=[
                GroupElement(
                    'lattice',
                    element=CyclicSpeciesElement,
                    space=LatticeSpace(tuple(self.shape)),
                    values={'strategy': strategies},
                    on='sites',
                ),
            ],
            interactions=[
                LocalRuleInteraction(
                    'invasion',
                    target='lattice',
                    state='strategy',
                    rule=self._invade,
                    neighborhood=NEAREST,
                    schedule='random-sequential',
                    stochastic=True,
                ),
            ],
        )
