"""Base of the solvers of the compiler's problem, which run its events, renewals, delays, and boundary laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import warnings
from collections.abc import Callable

import numpy as np

from ...errors import SolverError
from ...types import Array
from ..problems import Problem
from ._base import BaseSolver
from ._event_locator import EventLocator
from ._history import _RESTART, History
from ._renewals import RenewalClock


class Solver(BaseSolver[Problem]):
    """Base of the solvers of the compiler's problem, which run its events, renewals, delays, and boundary laws.

    Every call to `run` receives the current parameters, so parameter changes apply at the next call, and applies the
    boundary laws of the problem after every internal step.
    """

    def _check_capabilities(self, problem: Problem) -> None:
        """Raise an error for renewal laws or fractional noise of a problem that the stepper lacks."""
        name = type(self).__name__
        if problem.renewals and not self._is_renewing():
            error_msg = (
                f'{name} does not fire renewal laws. Use RK4Solver for rate laws, EulerMaruyamaSolver for noise laws, '
                'or RenewalSolver for renewal laws alone.'
            )
            raise SolverError(error_msg)
        if problem.fractional and not self._is_fractional():
            laws = ', '.join(dict.fromkeys(rows.law for rows in problem.fractional))
            error_msg = (
                f'{name} draws independent Wiener increments, but the {laws} needs correlated increments. Use the '
                'fractional Euler-Maruyama stepper, FractionalEulerMaruyamaSolver.'
            )
            raise SolverError(error_msg)

    def _bounded(self, t: float, x: Array, p: Array) -> Array:
        """Return a state after the boundary laws of the problem, or the state itself without boundary laws."""
        boundary = self.problem_.boundary
        return x if boundary is None else boundary(t, x, p)

    def _advance_renewals(
        self,
        t: float,
        x: Array,
        p: Array,
        until: float,
        *,
        dt: float,
        step: Callable[[float, Array, Array, Array], Array],
    ) -> Array:
        """Integrate with fixed steps that stop at the renewals of every member and at the breakpoints of inputs.

        Members that do not interact take steps on their own clocks, and interacting members split every step at each
        renewal time inside it.
        """
        problem, clock = self.problem_, self.renewals_
        if problem.delays or problem.events or clock is None:
            error_msg = f'{type(self).__name__} does not combine renewal laws with delays or events in one system.'
            raise SolverError(error_msg)
        clock.begin(t, x, p)
        advance = clock.advance_split if problem.has_group_sums else clock.advance_members
        state, time = x.copy(), t
        for bound in EventLocator._find_bounds(problem, t, until):
            while time < bound - _RESTART * max(1.0, abs(bound)):
                end = min(time + dt, bound)
                state, time = advance(time, end, state, p, step), end
            time = bound
        return state

    def _warm(self, problem: Problem) -> None:
        """Call every numeric function of the problem once, so compiled backends finish compiling before a run."""
        x, p = self.initial_state_.copy(), problem.initial_parameters.copy()
        rng = np.random.default_rng(0)
        drawn = (rng,) if problem.stochastic else ()
        for function in (problem.drift, problem.diffusion, problem.propensities, problem.boundary):
            if function is not None:
                function(0.0, x, p)
        if problem.update is not None:
            problem.update(0.0, x, p, *drawn)
        accelerated = problem.accelerated
        if accelerated is None:
            return
        if accelerated.rk4 is not None:
            accelerated.rk4(0.0, x, p, 0.0, 1.0)
        if accelerated.velocity_verlet is not None:
            empty = np.zeros(0, dtype=np.int64)
            accelerated.velocity_verlet(0.0, x, p, 0.0, 1.0, empty, empty)
        if accelerated.euler_maruyama is not None:
            accelerated.euler_maruyama(0.0, x, p, 0.0, 1.0, rng)
        if accelerated.gillespie is not None:
            accelerated.gillespie(0.0, x, p, 0.0, rng)
        if accelerated.iterate is not None:
            accelerated.iterate(0.0, x, p, 0.0, *drawn)

    def _prepare(self, problem: Problem) -> None:
        """Prepare the stepper for a problem and call its numeric functions once, so compiled backends compile.

        Args:
            problem:
                The compiled problem.

        Raises:
            SolverError:
                The problem has a form the stepper does not solve, or renewal laws or fractional noise it does not
                integrate.
        """
        super()._prepare(problem)
        self._check_capabilities(problem)
        self.locator_ = EventLocator(problem)
        self.renewals_ = RenewalClock(problem, self.rng_, self.locator_) if problem.renewals else None
        self.history_: History | None = None
        step = getattr(self, 'dt', None)
        limit = problem.stable_step
        if limit is not None and step is not None and step > limit and not self._is_implicit():
            warning_msg = (
                f'{type(self).__name__} takes steps of {step}, above the stability limit {limit:.6g} of the diffusion '
                f"of the fields. Use dt <= {limit:.6g}, ScipyODESolver(method='BDF') or ScipyODESolver(method='Radau') "
                'for finite-difference and finite element fields, or a split stepper such as ETDRK4Solver for '
                'spectral fields.'
            )
            warnings.warn(warning_msg, UserWarning, stacklevel=2)
        self._warm(problem)

    def _collect(self) -> None:
        """Set the events recorded since the last run, and forget them, and the time of a terminal event."""
        locator = self.locator_
        self.events_, locator.records = locator.records, []
        self.stopped_at_ = locator.stopped_at
