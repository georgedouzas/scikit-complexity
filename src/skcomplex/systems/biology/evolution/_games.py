"""Evolution systems: evolutionary games of strategies that spread with their payoff."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement, LatticeElement
from ....core.components.interactions import LocalRuleInteraction
from ....core.components.laws.utils import total
from ....core.components.spaces import LatticeSpace, VonNeumannNeighborhood
from ....core.components.system import System
from ....core.specs import StateSpec, VisualSpec
from ....core.types import Array
from ..._catalog import register_system
from ._replicator import ReplicatorInteraction

NEAREST = VonNeumannNeighborhood()

STRATEGIES = 3


class CyclicSpeciesElement(Element):
    """Site of a lattice of rock, paper, and scissors, 0, 1, and 2, where each strategy beats the one before it.

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
    """Strategy of an evolutionary game, with the share `x` of the population that plays it.

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


def _invade(strategy: Array, *around: Array, random: Array) -> Array:
    """Return the strategy of every site after a random neighbour that beats it invades it."""
    choice = np.minimum((random * len(around)).astype(int), len(around) - 1)
    neighbor = np.choose(choice, np.stack(around))
    beaten = neighbor == np.mod(strategy + 1, STRATEGIES)
    return np.where(beaten, neighbor, strategy)


@register_system(
    field='biology/evolution',
    tags=('evolutionary game', 'replicator', 'ode', 'game theory'),
    references=('Taylor and Jonker 1978', 'Maynard Smith 1982', 'Gros 2015', 'Szabó and Fáth 2007'),
    defaults={'until': 30.0, 'dt': 0.1},
)
def replicator_dynamics(
    x: Sequence[float] | None = None,
    *,
    payoff: Sequence[Sequence[float]] | None = None,
    V: float = 2.0,
    C: float = 3.0,
) -> System:
    """Return the replicator dynamics of a game, by default hawks and doves, which settle at a share `V / C` of hawks.

    Every strategy is a member of the group `population`, and it grows with its payoff above the mean payoff. The
    hawks and doves game has the payoff matrix `((V - C) / 2, V), (0, V / 2)` with hawks first. When the cost of a fight
    `C` exceeds the value `V`, hawks and doves coexist with `V / C` hawks, and otherwise hawks take over. A payoff
    matrix gives any other game, such as the prisoner's dilemma or rock, paper, scissors.

    Args:
        x:
            The initial shares of the strategies, or `None` for equal shares.

        payoff:
            The payoff matrix of another game, one row per strategy, or `None` for hawks and doves.

        V:
            The value of the resource contested by hawks and doves.

        C:
            The cost of a fight between hawks.

    Returns:
        system:
            The system, with the shares `population.x` of the strategies, hawks first.

    Examples:
        >>> from skcomplex.systems.biology.evolution import replicator_dynamics
        >>> replicator_dynamics(payoff=((0, -1, 1), (1, 0, -1), (-1, 1, 0))).elements[0].prepare().size_
        3
    """
    matrix = (((V - C) / 2, V), (0.0, V / 2)) if payoff is None else tuple(tuple(row) for row in payoff)
    strategies = len(matrix)
    shares = [1 / strategies] * strategies if x is None else list(x)
    population = GroupElement('population', element=StrategyElement, size=strategies, values={'x': shares})
    weights = np.asarray(matrix, dtype=float)
    replication = ReplicatorInteraction(
        'replication',
        target='population',
        fitness=lambda s: total(s.target.x, weights=weights),
    )
    return System(elements=[population], interactions=[replication])


@register_system(
    field='biology/evolution',
    tags=('evolutionary game', 'lattice', 'cyclic dominance', 'spirals', 'stochastic'),
    references=('Szabó and Fáth 2007', 'Reichenbach, Mobilia, and Frey 2007'),
    defaults={'until': 100, 'dt': 1},
)
def rock_paper_scissors(
    shape: Sequence[int] = (60, 60),
    *,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return rock, paper, and scissors on a lattice, whose cyclic invasions keep the three strategies coexisting.

    Starting from random strategies, the strategies chase each other in domains and fronts, and each keeps about a
    third of the lattice.

    Args:
        shape:
            The number of rows and columns of the periodic lattice.

        random_state:
            The seed or generator of the initial strategies.

    Returns:
        system:
            The system, with the strategies `lattice.strategy`.

    Examples:
        >>> from skcomplex.systems.biology.evolution import rock_paper_scissors
        >>> rock_paper_scissors((10, 10)).elements[0].prepare().size_
        100
    """
    strategies = np.random.default_rng(random_state).integers(0, STRATEGIES, size=tuple(shape)).astype(float)
    return System(
        elements=[
            LatticeElement(
                'lattice',
                element=CyclicSpeciesElement,
                space=LatticeSpace(tuple(shape)),
                values={'strategy': strategies},
            ),
        ],
        interactions=[
            LocalRuleInteraction(
                'invasion',
                target='lattice',
                state='strategy',
                rule=_invade,
                neighborhood=NEAREST,
                schedule='random-sequential',
                stochastic=True,
            ),
        ],
    )
