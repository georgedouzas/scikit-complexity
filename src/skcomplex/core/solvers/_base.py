"""Define the stepper contract and the registry of solvers per form."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import warnings
from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np
from sklearn.base import BaseEstimator

from ..base import Form, Registry, SolverError, Supports
from ..compile import EventRecord, ExtremalRecord, Problem, RelaxationRecord, RenewalRecord, Vector
from ._hybrid import EventLocator, History
from ._renewals import RenewalClock


class Stepper(BaseEstimator):
    """Base class of solvers that advance a problem of one form through time.

    Subclasses set the class variable `form` and implement `advance`. Every call to `advance` receives the current
    parameters, so parameter changes apply at the next call, and applies the boundary laws of the problem after every
    internal step. Steppers that treat diffusion implicitly set `implicit`,
    so they do not warn about the stability limit of explicit steps. Steppers that fire renewal laws set `renews`, and
    steppers that integrate fractional noise set `fractional`.
    """

    supports: ClassVar[Supports]

    form: ClassVar[Form]
    implicit: ClassVar[bool] = False
    renews: ClassVar[bool] = False
    fractional: ClassVar[bool] = False

    def _check_capabilities(self, problem: Problem) -> None:
        """Raise an error when a problem has renewal laws or fractional noise that the stepper does not integrate."""
        name = type(self).__name__
        if problem.renewals and not self.renews:
            error_msg = (
                f'{name} does not fire renewal laws. Use RK4 for rate laws, EulerMaruyama for noise laws, or '
                'RenewalStepper for renewal laws alone.'
            )
            raise SolverError(error_msg)
        if problem.fractional and not self.fractional:
            laws = ', '.join(dict.fromkeys(rows.law for rows in problem.fractional))
            error_msg = (
                f'{name} draws independent Wiener increments, but the {laws} needs correlated increments. Use the '
                'fractional Euler-Maruyama stepper, FractionalEulerMaruyama.'
            )
            raise SolverError(error_msg)

    def _bounded(self, t: float, x: Vector, p: Vector) -> Vector:
        """Return a state after the boundary laws of the problem, or the state itself without boundary laws."""
        boundary = self.problem_.boundary
        return x if boundary is None else boundary(t, x, p)

    def initialize(self, problem: Problem, rng: np.random.Generator) -> None:
        """Prepare the stepper for a problem.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator of stochastic solvers.

        Raises:
            SolverError:
                The problem has a form the stepper does not solve, or renewal laws or fractional noise it does not
                integrate.
        """
        if problem.form is not self.form:
            error_msg = (
                f'{type(self).__name__} solves the {self.form.value} form, but the problem has the '
                f'{problem.form.value} form.'
            )
            raise SolverError(error_msg)
        self._check_capabilities(problem)
        self.problem_ = problem
        self.rng_ = rng
        self.locator_ = EventLocator(problem)
        self.renewals_ = RenewalClock(problem, rng, self.locator_) if problem.renewals else None
        self.history_: History | None = None
        step = getattr(self, 'dt', None)
        limit = getattr(problem, 'stable_step', None)
        if limit is not None and step is not None and step > limit and not self.implicit:
            warning_msg = (
                f'{type(self).__name__} takes steps of {step}, above the stability limit {limit:.6g} of the diffusion '
                f"of the fields. Use dt <= {limit:.6g}, ScipyODE(method='BDF') or ScipyODE(method='Radau') for "
                'finite-difference and finite element fields, or a split stepper such as ETDRK4 for spectral fields.'
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

    def plan(self, until: float) -> None:
        """Prepare for a run that ends at a time, which steppers that draw the noise of a whole run need.

        Args:
            until:
                The end time of the run, infinite for a run that goes on until it is stopped.
        """
        del until

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

    def drain_events(self) -> list[EventRecord | RelaxationRecord | RenewalRecord | ExtremalRecord]:
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
