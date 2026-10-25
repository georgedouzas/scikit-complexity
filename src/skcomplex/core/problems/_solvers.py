"""Define the base of the solvers, which advance a compiled problem through time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from ..base import BaseEstimator, Form, SolverError, Supports, Vector
from ._problem import BaseProblem


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

    supports: ClassVar[Supports]

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
