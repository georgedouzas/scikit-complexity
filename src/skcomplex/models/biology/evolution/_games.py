"""Evolution models: evolutionary games of strategies that spread with their payoff."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, Expression, role, state
from ....core.model import (
    Element,
    Interaction,
    LatticeSpace,
    Law,
    LocalRule,
    Neighborhood,
    Rate,
    Symbols,
    System,
    total,
)
from ....core.runtime import register_model

type Array = npt.NDArray[np.float64]

NEAREST = Neighborhood.von_neumann()
STRATEGIES = 3


def _replicator_rate(x: Expression, payoff: Array) -> Expression:
    """Return the replicator rate `x_i ((A x)_i - x . A x)` of a strategy that reads the payoff matrix as weights."""
    fitness = total(x, weights=payoff)
    return x * (fitness - total(x * fitness))


def _invade(strategy: Array, *around: Array, random: Array) -> Array:
    """Return the strategy of every site after a random neighbour that beats it invades it."""
    choice = np.minimum((random * len(around)).astype(int), len(around) - 1)
    neighbor = np.choose(choice, np.stack(around))
    beaten = neighbor == np.mod(strategy + 1, STRATEGIES)
    return np.where(beaten, neighbor, strategy)


Strategy = Element.define('Strategy', states={'x': state(default=0.5, domain=Domain.NONNEGATIVE)})


class Replication(Interaction):
    """Replicator equation of the strategies of a group, `dx_i/dt = x_i ((A x)_i - x . A x)`, with a payoff matrix `A`.

    Every strategy is a member of the group, and its fitness `(A x)_i` is a total over the strategies weighted by its
    row of the payoff matrix. The mean fitness `x . A x` is the total of the shares times their fitness.

    Args:
        label:
            The label of the interaction.

        population:
            The path of the group of strategies.

        payoff:
            The payoff matrix, one row per strategy.

    Examples:
        >>> from skcomplex.models.biology.evolution import Replication
        >>> Replication('replication', population='players', payoff=((0, 1), (1, 0))).payoff
        ((0, 1), (1, 0))
    """

    population: str = role()
    payoff: Sequence[Sequence[float]] = ((0.0, 0.0), (0.0, 0.0))

    def laws(self, s: Symbols) -> list[Law]:
        """Return the replicator equation of the payoff matrix.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        payoff = np.asarray(self.payoff, dtype=float)
        return [Rate(s.population.x, _replicator_rate(s.population.x, payoff))]


class CyclicSpecies(Element):
    """Site of a lattice of rock, paper, and scissors, 0, 1, and 2, where each strategy beats the one before it.

    Sites update one at a time in random order, and a site adopts the strategy of a random neighbour that beats it.

    Args:
        label:
            The label of the site or the lattice group.

        strategy:
            The initial strategy, 0, 1, or 2.

    Examples:
        >>> from skcomplex.models.biology.evolution import CyclicSpecies
        >>> CyclicSpecies('site', strategy=2).strategy
        2
    """

    strategy: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the stochastic invasion by a random neighbour.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [LocalRule(s.strategy, NEAREST, _invade, schedule='random-sequential', stochastic=True)]


@register_model(
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
            The model, with the shares `population.x` of the strategies, hawks first.

    Examples:
        >>> from skcomplex.models.biology.evolution import replicator_dynamics
        >>> replicator_dynamics(payoff=((0, -1, 1), (1, 0, -1), (-1, 1, 0))).elements[0].size
        3
    """
    matrix = (((V - C) / 2, V), (0.0, V / 2)) if payoff is None else tuple(tuple(row) for row in payoff)
    strategies = len(matrix)
    shares = [1 / strategies] * strategies if x is None else list(x)
    population = Strategy.group('population', strategies, x=shares)
    replication = Replication('replication', population='population', payoff=matrix)
    return System(components=[population], interactions=[replication])


@register_model(
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
            The model, with the strategies `lattice.strategy`.

    Examples:
        >>> from skcomplex.models.biology.evolution import rock_paper_scissors
        >>> rock_paper_scissors((10, 10)).elements[0].size
        100
    """
    strategies = np.random.default_rng(random_state).integers(0, STRATEGIES, size=tuple(shape)).astype(float)
    return System(components=[CyclicSpecies.lattice('lattice', LatticeSpace(tuple(shape)), strategy=strategies)])
