"""Urn systems.

The module holds the Ehrenfest system of balls moved one at a time between two boxes.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.laws.utils import bernoulli
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import Role
from ....core.systems import BaseSystem, System


class EhrenfestMoveInteraction(Interaction):
    """Move of one ball drawn at random between the two boxes of the Ehrenfest urn.

    The count `n` of the first box falls by one with probability `n / N` and rises by one otherwise.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the urn, whose value `n` counts the balls of the first box.

        N:
            The number of balls.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import EhrenfestMoveInteraction
        >>> EhrenfestMoveInteraction('move', target='urn', N=10).N
        10
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role, N: Annotated[float, RealSet.open(0, oo)] = 40) -> None:
        super().__init__(label)
        self.target = target
        self.N = N

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the move of one ball, out of the first box with probability `n / N` and into it otherwise.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            n = context.target.n
            self.laws_ = [UpdateLaw(n, n + 1 - 2 * bernoulli('pick', n / context.N))]


class EhrenfestUrnSystem(BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import EhrenfestUrnSystem
        >>> trajectory = EhrenfestUrnSystem(10).prepare().simulate(until=1, random_state=0)
        >>> int(trajectory.get_values('urn.n')[0])
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_balls = n_balls
        self.in_first = in_first

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the urn and the move of its balls.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        n = self.n_balls if self.in_first is None else self.in_first
        self.system_ = System(
            self.label,
            elements=[Element('urn', values={'n': n})],
            interactions=[EhrenfestMoveInteraction('move', target='urn', N=self.n_balls)],
        )
