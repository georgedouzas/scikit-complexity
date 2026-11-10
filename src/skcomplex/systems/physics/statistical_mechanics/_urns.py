"""Urn systems.

The module holds the Ehrenfest system of balls moved one at a time between two boxes.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.laws.utils import bernoulli
from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


class EhrenfestUrnSystem(DynamicalSystem):
    """The Ehrenfest urn system.

    Balls move one at a time between two boxes, and the system relaxes to equal counts.

    At every step one of the `N` numbered balls is drawn at random and moved to the other box, so the count `n` of the
    first box falls by one with probability `n / N` and rises by one otherwise. Started with all balls in the first box,
    the count relaxes to `N / 2` with the exponential mean `N / 2 + (n_0 - N / 2) (1 - 2 / N)^t`, and then fluctuates
    around it: the chain is reversible, yet the relaxation looks irreversible. Its stationary distribution is the
    binomial distribution of `N` balls with probability one half. The system has the count `urn.n` of balls in the first
    box.

    Args:
        n_balls:
            The number of balls `N`.

        in_first:
            The initial number of balls in the first box, or `None` for all of them.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import EhrenfestUrnSystem
        >>> EhrenfestUrnSystem(10).prepare().system_.elements[0].get_value('n')
        10
    """

    field = 'physics/statistical_mechanics'
    topics = ('markov chain', 'irreversibility', 'entropy', 'discrete map')
    references = ('Ehrenfest and Ehrenfest 1907', 'Barsky 2020', 'Kac 1947')
    until = 200
    dt = 1

    def __init__(
        self,
        n_balls: int = 40,
        *,
        in_first: int | None = None,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.n_balls = n_balls
        self.in_first = in_first

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'urn'

    def _get_time(self) -> str:
        """Return the discrete time of the map."""
        return 'discrete'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the count of the first box and the number of balls with their specs."""
        n = self.n_balls if self.in_first is None else self.in_first
        return {
            'n': DeclaredVariableSpec(n, StateSpec(domain=NN)),
            'N': DeclaredVariableSpec(self.n_balls, ParameterSpec(domain=RealSet.open(0, oo))),
        }

    def _get_equations(self) -> Equations:
        """Return the move of one ball, out of the box with probability `n / N` and into it otherwise."""
        return {'n': lambda s: s.target.n + 1 - 2 * bernoulli('pick', s.target.n / s.target.N)}
