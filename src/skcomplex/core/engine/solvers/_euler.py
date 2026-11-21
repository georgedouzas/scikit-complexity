"""Explicit Euler integrator of ordinary differential equations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np

from ...components.laws import EventLaw, RateLaw
from ...components.laws._boundary import _BoundaryLaw
from ...components.spaces._continuous_time import _ContinuousTime
from ...enums import FormKind, OrderKind
from ...errors import SolverError
from ...types import Array
from ..backends import SupportSpec
from ..problems._steps import step_count
from ..types import Interpolant
from ._base import BaseSolver
from ._history import History
from ._solver import Solver


class EulerSolver(Solver):
    """A solver that steps ordinary differential equations by the explicit Euler rule.

    Every step of length `h` sets `x + h f(t, x)`, so a population that doubles its rate grows from 1 to 2.25 over one
    time unit in two steps of 0.5. It is the rule that Vensim and Stella use by default, so a run with the time step of
    a model file of those tools matches it step by step. Choose it for teaching and for quick checks with a small step,
    in models of populations, stocks and flows, and cooling.

    Read more in the user guide, under Solvers.

    Args:
        dt:
            The largest internal time step.

        random_state:
            The seed or generator of stochastic solvers.

        until:
            The end time of the run the solver prepares for, infinite for a run that goes on until it is stopped.

    See Also:
        RK4Solver: Steps with the classical Runge-Kutta rule of fourth order.
        ScipyODESolver: Steps with adaptive steps of SciPy.

    Examples:
        >>> from skcomplex.core.engine.solvers import EulerSolver
        >>> from skcomplex.systems.biology.ecology import ExponentialGrowthSystem
        >>> system = ExponentialGrowthSystem(n=1.0, r=1.0, solver=EulerSolver(dt=0.5)).prepare()
        >>> system.simulate(until=1, dt=1).get_values('population.x').tolist()
        [1.0, 2.25]
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={_ContinuousTime},
            space={None},
            laws={RateLaw, EventLaw, _BoundaryLaw},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.ODE

    def __init__(
        self,
        dt: float = 1e-3,
        *,
        random_state: int | np.random.Generator | None = None,
        until: float = math.inf,
    ) -> None:
        self.dt = dt
        self.random_state = random_state
        self.until = until

    def _advance_hybrid(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Integrate with fixed Euler steps.

        Every step is interpolated for delays, and events are located in it.
        """
        problem = self.problem_
        drift = self._get_drift()
        history = self.history_ = History._open_history(problem, self.history_, t, x)

        def take(time: float, state: Array, step: float, ceiling: float) -> tuple[Array, Interpolant]:
            """Return the state after one step and its straight line over the step."""
            start = min(time, ceiling)
            slope = drift(start, state, History._copy_with_delays(problem, history, start, state, p))
            following = state + step * slope

            def interpolant(moment: float) -> Array:
                """Return the state at a moment of the step."""
                return state + (moment - time) * slope

            return following, interpolant

        return BaseSolver._advance_fixed_steps(
            t,
            x,
            p,
            until,
            dt=self.dt,
            locator=self.locator_,
            history=history,
            bounded=self._get_bounded,
            take=take,
        )

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Integrate the drift from time `t` to time `until` in equal steps.

        The steps are no longer than `dt`.

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

        Raises:
            SolverError:
                The internal step is not positive.
        """
        if not self.dt > 0:
            error_msg = f'dt = {self.dt!r} must be positive.'
            raise SolverError(error_msg)
        if until <= t or self.problem_.drift is None:
            return x
        if BaseSolver._is_hybrid(self.problem_, t, until):
            return self._advance_hybrid(t, x, p, until)
        drift = self._get_drift()
        n_steps = step_count(t, until, self.dt)
        step = (until - t) / n_steps
        state = x.copy()
        for index in range(n_steps):
            time = t + index * step
            state = self._get_bounded(time + step, state + step * drift(time, state, p), p)
        return state
