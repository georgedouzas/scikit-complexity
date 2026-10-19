"""Define the stepper contract and the registry of solvers per form."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import warnings
from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np
from sklearn.base import BaseEstimator

from ..compile import EventRecord, Problem, RelaxationRecord, Vector
from ..core import Form, Registry, SolverError
from ._hybrid import EventLocator, History


class Stepper(BaseEstimator):
    """Base class of solvers that advance a problem of one form through time.

    Subclasses set the class variable `form` and implement `advance`. Every call to `advance` receives the current
    parameters, so parameter changes apply at the next call.
    """

    form: ClassVar[Form]

    def initialize(self, problem: Problem, rng: np.random.Generator) -> None:
        """Prepare the stepper for a problem.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator of stochastic solvers.

        Raises:
            SolverError:
                The problem has a form the stepper does not solve.
        """
        if problem.form is not self.form:
            error_msg = (
                f'{type(self).__name__} solves the {self.form.value} form, but the problem has the '
                f'{problem.form.value} form.'
            )
            raise SolverError(error_msg)
        self.problem_ = problem
        self.rng_ = rng
        self.locator_ = EventLocator(problem)
        self.history_: History | None = None
        step = getattr(self, 'dt', None)
        limit = getattr(problem, 'stable_step', None)
        if limit is not None and step is not None and step > limit:
            warning_msg = (
                f'{type(self).__name__} takes steps of {step}, above the stability limit {limit:.6g} of the diffusion '
                f'of the fields. Use dt <= {limit:.6g}, or an adaptive or implicit solver.'
            )
            warnings.warn(warning_msg, UserWarning, stacklevel=2)

    @abstractmethod
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

    def initial_state(self, x: Vector) -> Vector:
        """Return the state vector of the first frame, which steppers that derive states can fill in.

        Args:
            x:
                The initial state vector of the problem.

        Returns:
            state:
                The state vector of the first frame.
        """
        return x

    def drain_events(self) -> list[EventRecord | RelaxationRecord]:
        """Return the events recorded since the last call, and forget them.

        Returns:
            events:
                The events in the order they happened.
        """
        locator = getattr(self, 'locator_', None)
        if locator is None:
            return []
        records, locator.records = locator.records, []
        return records

    @property
    def stopped_at(self) -> float | None:
        """Time at which a terminal event ended the run, or `None` while it goes on."""
        locator = getattr(self, 'locator_', None)
        return None if locator is None else locator.stopped_at

    def agents_of(self, group: str) -> list[Any] | None:
        """Return the agents of an agent group by path, or `None` when the stepper holds no such group.

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
        """Return the state the stepper keeps outside the state vector, such as agents, or `None`.

        Returns:
            snapshot:
                The external state at the current time, or `None` for steppers that keep none.
        """
        return None


SOLVERS = Registry[type[Stepper]]('solver')


def register_solver(name: str, stepper: type[Stepper], *, default: bool = False) -> None:
    """Register a stepper class under a name for its form.

    Args:
        name:
            The name users select the stepper with.

        stepper:
            The stepper class.

        default:
            Whether the stepper is the default of its form.
    """
    SOLVERS.register((stepper.form, name), stepper)
    if default:
        SOLVERS.register((stepper.form, 'auto'), stepper)


def resolve_solver(problem: Problem, solver: str | Stepper) -> Stepper:
    """Return a stepper for a problem from a registered name or an instance.

    Args:
        problem:
            The compiled problem.

        solver:
            A registered name, `'auto'` for the default of the problem's form, or a stepper instance.

    Returns:
        stepper:
            A new stepper for a name, or the given instance.

    Raises:
        SolverError:
            No stepper of the problem's form is registered under the name.
    """
    if isinstance(solver, Stepper):
        return solver
    try:
        return SOLVERS.resolve((problem.form, solver))()
    except KeyError as error:
        names = [str(key[1]) for key in SOLVERS if isinstance(key, tuple) and key[0] is problem.form]
        error_msg = f'No {problem.form.value} solver named {solver!r}. Available names are: {", ".join(names)}.'
        raise SolverError(error_msg) from error
