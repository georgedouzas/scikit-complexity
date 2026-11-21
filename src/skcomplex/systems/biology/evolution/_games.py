"""Evolution systems: evolutionary games.

The strategies spread with their payoff.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.laws.utils import total
from ....core.components.spaces import BaseSpace, LatticeSpace, RadiusNeighborhood
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ..common.populations import ReplicatorInteraction

NEAREST = RadiusNeighborhood(1, metric='manhattan')
STRATEGIES = 3


class CyclicSpeciesElement(Element):
    """Site of a lattice of rock, paper, and scissors, 0, 1, and 2.

    Each strategy beats the one before it.

    The sites update one at a time in random order by a `CyclicInvasionInteraction`, and a site adopts the strategy of a
    random neighbour that beats it.

    Args:
        label:
            The label of the site or the lattice group.

        strategy:
            The initial strategy, 0, 1, or 2, one per site or shared.

        space:
            The lattice whose sites hold one strategy each, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.biology.evolution import CyclicSpeciesElement
        >>> CyclicSpeciesElement('site', strategy=2).strategy
        2
    """

    def __init__(
        self,
        label: str,
        *,
        strategy: Annotated[float | Array, NN] = 0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.strategy = strategy


class StrategyElement(Element):
    """Strategy of an evolutionary game, with the share `x` of players.

    The share is the one of the population that plays the strategy.

    Args:
        label:
            The label of the strategy or of the group of strategies.

        x:
            The initial share, one per strategy or shared.

        size:
            The number of strategies, or `None` for one.

    Examples:
        >>> from skcomplex.systems.biology.evolution import StrategyElement
        >>> StrategyElement('hawks', x=0.4).x
        0.4
    """

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[object, RealSet.closed_open(0, oo)] = 0.5,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.evolution import ReplicatorDynamicsSystem
        >>> system = ReplicatorDynamicsSystem(payoff=((0, -1, 1), (1, 0, -1), (-1, 1, 0))).prepare().system_
        >>> system.elements[0].size
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.payoff = payoff
        self.V = V
        self.C = C

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of replicator dynamics of a game.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        matrix = (
            (((self.V - self.C) / 2, self.V), (0.0, self.V / 2))
            if self.payoff is None
            else tuple(tuple(row) for row in self.payoff)
        )
        strategies = len(matrix)
        shares = [1 / strategies] * strategies if self.x is None else list(self.x)
        population = StrategyElement('population', x=shares, size=strategies)
        weights = np.asarray(matrix, dtype=float)
        replication = ReplicatorInteraction(
            'replication',
            target='population',
            fitness=lambda s: total(s.target.x, weights=weights),
        )
        self.system_ = System(self.label, elements=[population], interactions=[replication])


class CyclicInvasionInteraction(Interaction):
    """Invasion of the sites of rock, paper, and scissors by a random neighbour that beats them.

    The sites update one at a time in a random order, and each adopts the strategy of a random von Neumann neighbour
    when that strategy beats its own, rock 0 beaten by paper 1, paper by scissors 2, and scissors by rock.

    Args:
        label:
            The label of the interaction.

        target:
            The sites whose state `strategy` the rule updates.

    Examples:
        >>> from skcomplex.systems.biology.evolution import CyclicInvasionInteraction
        >>> CyclicInvasionInteraction('invasion', target='lattice').target
        'lattice'
    """

    def __init__(self, label: str, *, target: Role = 'lattice') -> None:
        super().__init__(label)
        self.target = target

    @staticmethod
    def _invade(strategy: Array, *around: Array, random: Array) -> Array:
        """Return the strategy of every site after invasion.

        A random neighbour that beats a site invades it.
        """
        choice = np.minimum((random * len(around)).astype(int), len(around) - 1)
        neighbor = np.choose(choice, np.stack(around))
        beaten = neighbor == np.mod(strategy + 1, STRATEGIES)
        return np.where(beaten, neighbor, strategy)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the invasion of the sites as the law.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        strategy = context.target.strategy
        self.laws_: list[BaseComponent] = [
            UpdateLaw(strategy, self._invade, neighborhood=NEAREST, schedule='random-sequential', label=self.label),
        ]


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.evolution import RockPaperScissorsSystem
        >>> RockPaperScissorsSystem((10, 10)).prepare().system_.elements[0].strategy.shape
        (10, 10)
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of rock, paper, and scissors on a lattice.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        strategies = (
            np.random.default_rng(self.random_state).integers(0, STRATEGIES, size=tuple(self.shape)).astype(float)
        )
        self.system_ = System(
            self.label,
            elements=[
                CyclicSpeciesElement('lattice', strategy=strategies, space=LatticeSpace(tuple(self.shape))),
            ],
            interactions=[
                CyclicInvasionInteraction('invasion', target='lattice'),
            ],
        )
