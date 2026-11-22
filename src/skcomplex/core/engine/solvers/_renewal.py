"""Exact stepper of renewal processes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...components.laws import ConstraintLaw, JumpLaw, NoiseLaw, RateLaw
from ...components.laws._boundary import _BoundaryLaw
from ...components.spaces import BaseSpace
from ...components.spaces._continuous_time import _ContinuousTime
from ...enums import FormKind, NoiseKind, OrderKind
from ...types import Array
from ..backends import SupportSpec
from ._solver import Solver


class RenewalSolver(Solver):
    """A solver that runs renewal processes exactly, from renewal to renewal.

    Every member renews on its own clock, with waiting times of any law, such as a walker that rests for a heavy tailed
    time before each jump, and the solver jumps from one renewal to the next without a time grid. A frame holds the
    values that the last renewal at or before its time set. It runs walkers trapped in a glass, neurons that fire, and
    machines that fail and are repaired.

    Read more in the user guide, under Solvers.

    See Also:
        GillespieSolver: Runs jumps whose waiting times have no memory.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import ContinuousTimeRandomWalkSystem
        >>> system = ContinuousTimeRandomWalkSystem(n_walkers=3).prepare()
        >>> type(system.system_.solver_).__name__, system.simulate(until=10, dt=10).get_values('walkers.x').shape
        ('RenewalSolver', (2, 3, 2))
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={_ContinuousTime},
            space={None, BaseSpace},
            laws={JumpLaw, RateLaw, NoiseLaw, ConstraintLaw, _BoundaryLaw},
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
