"""Exact stepper of renewal processes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...components.laws import BoundaryLaw, ConstraintLaw, JumpLaw, NoiseLaw, RateLaw
from ...components.spaces import BaseSpace, ContinuousTime
from ...enums import FormKind, NoiseKind, OrderKind
from ...types import Array
from ..backends import SupportSpec
from ._solver import Solver


class RenewalSolver(Solver):
    """A solver that steps renewal processes exactly.

    It jumps from renewal to renewal without a time grid. Every member renews on its own clock, and a frame holds the
    states the last renewal at or before its time set.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.interactions import Interaction
        >>> from skcomplex.core.components.laws import JumpLaw
        >>> from skcomplex.core.components.laws.utils import draw
        >>> from skcomplex.core.engine.solvers import RenewalSolver, StepSpec
        >>> from skcomplex.core.systems import System
        >>> def count(s):
        ...     return [JumpLaw(waiting=draw('wait', 'expon'), reset={s.target.n: s.target.n + 1})]
        >>> counting = Interaction('counting', roles={'target': 'counters'}, laws=count)
        >>> system = System(elements=[Element('counters', values={'n': 0.0}, size=3)], interactions=[counting])
        >>> problem = system.prepare().simulate(until=0).problem
        >>> stepper = RenewalSolver(random_state=0).prepare(problem)
        >>> state = stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 100.0)).state_
        >>> bool(np.all(state[:3] > 50))
        True
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={ContinuousTime},
            space={None, BaseSpace},
            laws={JumpLaw, RateLaw, NoiseLaw, ConstraintLaw, BoundaryLaw},
            noise=frozenset({NoiseKind.DIAGONAL}),
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.RENEWAL

    @classmethod
    def _is_renewing(cls) -> bool:
        """Return whether the solver fires renewal laws."""
        return True

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Fire every renewal from time `t` up to time `until`, each member in time order.

        Args:
            t:
                The current time.

            x:
                The current state vector.

            p:
                The parameter vector.

            until:
                The time to advance to.

        Returns:
            state:
                The state vector at time `until`.
        """
        clock = self.renewals_
        if until <= t or clock is None:
            return x
        clock.begin(t, x, p)
        return self._get_bounded(until, clock.fire_until(until, x, p), p)
