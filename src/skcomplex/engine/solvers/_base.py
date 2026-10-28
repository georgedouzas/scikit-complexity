"""Define the base of the solvers, which advance a compiled problem through time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from ...core.errors import SolverError
from ...core.types import Vector
from ..problems import Problem
from ..problems._kinds import Form
from ..problems._problem import BaseProblem
from ..problems._support import SupportSpec
from ..utils import BaseEstimator
from ._hybrid import _RESTART, Bounded, EventLocator, History, Take


class BaseSolver[P: BaseProblem](BaseEstimator):
    """Base of the solvers, which advance a problem of one form through time.

    A solver is prepared for a problem with `prepare(problem, rng, until)` and advances a state with `run(t, x, p,
    until)`, which returns the state at time `until`. The last call `run(..., final=True)` releases what preparing
    acquired, such as an external process. Subclasses set the class variable `form` and implement `_advance`, and
    solvers that keep events, agents, or other state outside the state vector also implement `_collect`. Solvers that
    also solve other forms list them in `forms`. Solvers that treat diffusion implicitly set `implicit`, so they do not
    warn about the stability limit of explicit steps. Solvers that fire renewal laws set `renews`, solvers that
    integrate fractional noise set `fractional`, and solvers that read external inputs set `reads_inputs`. A solver is
    generic over the problem it advances, so one written for the compiler's problem reads its fields.

    Attributes:
        problem_:
            The problem the solver is prepared to.

        rng_:
            The random number generator of stochastic solvers.

        initial_state_:
            The state vector of the first frame, which solvers that derive states fill in.

        events_:
            The events of the last run, in the order they happened.

        stopped_at_:
            The time at which a terminal event ended the run, or `None` while it goes on.

        snapshot_:
            The state the solver keeps outside the state vector, such as agents, after the last run, or `None`.

        agents_:
            The views of the agents alive after the last run, as rules receive them, by path of their agent group.
    """

    supports: ClassVar[SupportSpec]

    form: ClassVar[Form]
    forms: ClassVar[frozenset[Form]] = frozenset()
    implicit: ClassVar[bool] = False
    renews: ClassVar[bool] = False
    fractional: ClassVar[bool] = False
    reads_inputs: ClassVar[bool] = False

    problem_: P
    rng_: np.random.Generator
    initial_state_: Vector
    events_: list[Any]
    stopped_at_: float | None
    snapshot_: object | None
    agents_: Mapping[str, list[Any]]

    def _prepare(self, problem: P, rng: np.random.Generator, until: float = math.inf) -> None:
        """Prepare the solver for a problem and a run that ends at a time.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator of stochastic solvers.

            until:
                The end time of the run, infinite for a run that goes on until it is stopped, which solvers that draw
                the noise of a whole run need.

        Raises:
            SolverError:
                The problem has a form the solver does not solve.
        """
        del until
        if problem.form != self.form and problem.form not in self.forms:
            error_msg = (
                f'{type(self).__name__} solves the {self.form.value} form, but the problem has the '
                f'{problem.form.value} form.'
            )
            raise SolverError(error_msg)
        self.problem_ = problem
        self.rng_ = rng
        self.initial_state_ = problem.initial_state.copy()
        self.events_ = []
        self.stopped_at_ = None
        self.snapshot_ = None
        self.agents_ = {}

    def _advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Advance the state from time `t` to time `until`.

        Args:
            t:
                The current time.

            x:
                The current state vector.

            p:
                The parameter vector.

            until:
                The time to advance to, later than `t`.

        Returns:
            state:
                The state vector at time `until`.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    def _collect(self) -> None:
        """Set the events, the stop time, the snapshot, and the agents after a run; the base keeps none."""
        self.events_ = []

    def _run(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Advance the state from time `t` to time `until` and collect what the run produced.

        A run that does not go forward in time returns the state unchanged, so a final run of no length releases the
        solver.

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
        state = self._advance(t, x, p, until) if until > t else x
        self._collect()
        return state

    @staticmethod
    def _is_hybrid(problem: Problem, t: float, until: float) -> bool:
        """Check whether a problem needs the hybrid path between two times: delays, events, or breakpoints.

        Args:
            problem:
                The compiled problem.

            t:
                The start time.

            until:
                The end time.

        Returns:
            hybrid:
                Whether the problem has delays, events, or breakpoints up to and at the end.
        """
        return bool(problem.delays or problem.events or problem.breakpoints.between(t, math.nextafter(until, math.inf)))

    @staticmethod
    def _reject_hybrid(problem: Problem, solver: str) -> None:
        """Raise an error when a solver without a hybrid path receives delays or events.

        Args:
            problem:
                The compiled problem.

            solver:
                The name of the solver.

        Raises:
            SolverError:
                The problem has delays or events.
        """
        if problem.delays or problem.events:
            error_msg = (
                f'{solver} does not integrate delays or events. Use ScipyODESolver or RK4Solver for ODEs and '
                'EulerMaruyamaSolver, MilsteinSolver, or HeunSolver for SDEs.'
            )
            raise SolverError(error_msg)

    @staticmethod
    def _advance_fixed_steps(
        t: float,
        x: Vector,
        p: Vector,
        until: float,
        *,
        dt: float,
        locator: EventLocator,
        history: History | None,
        bounded: Bounded,
        take: Take,
    ) -> Vector:
        """Integrate with fixed steps that stop at breakpoints, record every step for delays, and locate events in it.

        Args:
            t:
                The current time.

            x:
                The current state vector.

            p:
                The parameter vector.

            until:
                The time to advance to.

            dt:
                The largest step.

            locator:
                The event locator of the run.

            history:
                The history that delayed laws read, or `None` without delays.

            bounded:
                The boundary laws `bounded(t, x, p)`.

            take:
                The step `take(t, x, h, ceiling)` of a scheme, which returns the state after a step of length `h` and
                its interpolant over the step, and reads inputs no later than `ceiling`.

        Returns:
            state:
                The state vector at time `until`, or at a terminal event.

        Raises:
            SolverError:
                The step exceeds the smallest delay.
        """
        problem = locator.problem
        smallest = EventLocator._find_smallest_delay(problem)
        if dt > smallest:
            error_msg = f'dt = {dt} exceeds the smallest delay {smallest}. Take a smaller step.'
            raise SolverError(error_msg)
        state, time = x.copy(), t
        for bound in EventLocator._find_bounds(problem, t, until):
            ceiling = EventLocator._find_left_limit(bound)
            while time < bound - _RESTART * max(1.0, abs(bound)):
                step = min(dt, bound - time)
                following, interpolant = take(time, state, step, ceiling)
                if history is not None:
                    history.add(time, time + step, interpolant)
                hit = (
                    locator.crossing(
                        time,
                        time + step,
                        state,
                        following,
                        p=p,
                        interpolant=interpolant,
                        include_hidden=False,
                    )
                    if problem.events
                    else None
                )
                if hit is not None:
                    time, index = hit
                    if history is not None:
                        history.truncate(time)
                    state = bounded(time, locator.apply(time, index, interpolant(time), p), p)
                    if locator.stopped_at is not None:
                        return state
                    continue
                state, time = bounded(time + step, following, p), time + step
            time = bound
        return state
