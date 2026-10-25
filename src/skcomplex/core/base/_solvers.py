"""Define the base of the solvers, which advance a compiled problem through time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar

import numpy as np
from sklearn.base import BaseEstimator

from ._errors import SolverError
from ._kinds import Form, Supports
from ._problems import BaseProblem
from ._utils import Vector


class BaseSolver[P: BaseProblem](BaseEstimator):
    """Base of the solvers, which advance a problem of one form through time.

    Subclasses set the class variable `form` and implement `advance`. Solvers that also solve other forms list them in
    `forms`. Solvers that treat diffusion implicitly set `implicit`, so they do not warn about the stability limit of
    explicit steps. Solvers that fire renewal laws set `renews`, solvers that integrate fractional noise set
    `fractional`, and solvers that read external inputs set `reads_inputs`. A solver is generic over the problem it
    advances, so one written for the compiler's problem reads its fields.
    """

    supports: ClassVar[Supports]

    form: ClassVar[Form]
    forms: ClassVar[frozenset[Form]] = frozenset()
    implicit: ClassVar[bool] = False
    renews: ClassVar[bool] = False
    fractional: ClassVar[bool] = False
    reads_inputs: ClassVar[bool] = False

    def initialize(self, problem: P, rng: np.random.Generator) -> None:
        """Prepare the solver for a problem.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator of stochastic solvers.

        Raises:
            SolverError:
                The problem has a form the solver does not solve.
        """
        if problem.form is not self.form and problem.form not in self.forms:
            error_msg = (
                f'{type(self).__name__} solves the {self.form.value} form, but the problem has the '
                f'{problem.form.value} form.'
            )
            raise SolverError(error_msg)
        self.problem_ = problem
        self.rng_ = rng

    def advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Advance the state from time `t` to time `until`.

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
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    def plan(self, until: float) -> None:
        """Prepare for a run that ends at a time, which solvers that draw the noise of a whole run need.

        Args:
            until:
                The end time of the run, infinite for a run that goes on until it is stopped.
        """
        del until

    def initial_state(self, x: Vector) -> Vector:
        """Return the state vector of the first frame, which solvers that derive states can fill in.

        Args:
            x:
                The initial state vector of the problem.

        Returns:
            state:
                The state vector of the first frame.
        """
        return x

    def drain_events(self) -> list[Any]:
        """Return the events recorded since the last call, and forget them, none by default.

        Returns:
            events:
                The events in the order they happened.
        """
        return []

    @property
    def stopped_at(self) -> float | None:
        """Time at which a terminal event ended the run, or `None` while it goes on."""
        return None

    def agents_of(self, group: str) -> list[Any] | None:
        """Return the agents of an agent group by path, or `None` when the solver holds no such group.

        Args:
            group:
                The path of the agent group.

        Returns:
            agents:
                The agents alive now, or `None`.
        """
        del group
        return None

    def snapshot(self) -> object | None:
        """Return the state the solver keeps outside the state vector, such as agents, or `None`.

        Returns:
            snapshot:
                The external state at the current time, or `None` for solvers that keep none.
        """
        return None
