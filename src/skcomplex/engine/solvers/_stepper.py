"""Define the stepper contract, the count of fixed steps, and the registry of solvers per form."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
import warnings
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from numba.extending import register_jitable

from ...core.base import BackendError, Form, LawKind, SolverError, Vector
from ...core.problems import BaseProblem, BaseSolver, Problem
from ..compile import SOLVERS, solver_registry
from ._hybrid import RESTART, EventLocator, History, find_bounds
from ._renewals import RenewalClock

STEP_TOLERANCE = 1e-9


@register_jitable
def step_count(t: float, until: float, dt: float) -> int:
    """Return the number of equal steps no longer than `dt` from time `t` to time `until`, at least one.

    A span within `1e-9` steps above a whole number of steps takes that whole number, so rounding in the span does not
    add a step. The compiled loops of the Numba backend call the same function.

    Args:
        t:
            The start time.

        until:
            The end time.

        dt:
            The largest step.

    Returns:
        count:
            The number of steps.

    Examples:
        >>> from skcomplex.engine.solvers import step_count
        >>> step_count(0.1, 0.4, 0.1)
        3
    """
    return max(1, math.ceil((until - t) / dt - STEP_TOLERANCE))


class Solver(BaseSolver[Problem]):
    """Base of the solvers of the compiler's problem, which run its events, renewals, delays, and boundary laws.

    Every call to `run` receives the current parameters, so parameter changes apply at the next call, and applies
    the boundary laws of the problem after every internal step.
    """

    def _check_capabilities(self, problem: Problem) -> None:
        """Raise an error for renewal laws, fractional noise, or objective laws of a problem that the stepper lacks."""
        name = type(self).__name__
        if problem.objectives and LawKind.OBJECTIVE not in self.supports.laws:
            error_msg = (
                f'{name} follows the laws of a system but does not optimise its objective laws. Use an optimal control '
                'solver, PontryaginSolver or CollocationSolver.'
            )
            raise SolverError(error_msg)
        if problem.renewals and not self.renews:
            error_msg = (
                f'{name} does not fire renewal laws. Use RK4Solver for rate laws, EulerMaruyamaSolver for noise laws, '
                'or RenewalSolver for renewal laws alone.'
            )
            raise SolverError(error_msg)
        if problem.fractional and not self.fractional:
            laws = ', '.join(dict.fromkeys(rows.law for rows in problem.fractional))
            error_msg = (
                f'{name} draws independent Wiener increments, but the {laws} needs correlated increments. Use the '
                'fractional Euler-Maruyama stepper, FractionalEulerMaruyamaSolver.'
            )
            raise SolverError(error_msg)

    def _bounded(self, t: float, x: Vector, p: Vector) -> Vector:
        """Return a state after the boundary laws of the problem, or the state itself without boundary laws."""
        boundary = self.problem_.boundary
        return x if boundary is None else boundary(t, x, p)

    def _advance_renewals(
        self,
        t: float,
        x: Vector,
        p: Vector,
        until: float,
        *,
        dt: float,
        step: Callable[[float, Vector, Vector, Vector], Vector],
    ) -> Vector:
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
        for bound in find_bounds(problem, t, until):
            while time < bound - RESTART * max(1.0, abs(bound)):
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

    def _prepare(self, problem: Problem, rng: np.random.Generator, until: float = math.inf) -> None:
        """Prepare the stepper for a problem and call its numeric functions once, so compiled backends compile.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator of stochastic solvers.

            until:
                The end time of the run.

        Raises:
            SolverError:
                The problem has a form the stepper does not solve, or renewal laws or fractional noise it does not
                integrate.
        """
        super()._prepare(problem, rng, until)
        self._check_capabilities(problem)
        self.locator_ = EventLocator(problem)
        self.renewals_ = RenewalClock(problem, rng, self.locator_) if problem.renewals else None
        self.history_: History | None = None
        step = getattr(self, 'dt', None)
        limit = problem.stable_step
        if limit is not None and step is not None and step > limit and not self.implicit:
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


def register_solver(name: str, stepper: type[Solver], *, default: bool = False) -> None:
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


def resolve_solver(problem: BaseProblem, solver: str | BaseSolver[Any]) -> BaseSolver[Any]:
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
    if isinstance(solver, BaseSolver):
        return solver
    solver_registry()
    try:
        return SOLVERS.resolve((problem.form, solver))()
    except KeyError as error:
        names = [str(key[1]) for key in SOLVERS.keys if isinstance(key, tuple) and key[0] == problem.form]
        error_msg = f'No {problem.form.value} solver named {solver!r}. Available names are: {", ".join(names)}.'
        raise SolverError(error_msg) from error


class ExternalSolver(Solver):
    """Base class of solvers that translate a compiled problem into a third-party library.

    Subclasses set the class variables `name` and `forms`, and implement `_unsupported`, `_build`, and `_advance`.
    Before the first frame, `prepare` checks every element, interaction, and law, and raises one error that lists
    everything the library cannot run.
    """

    name: ClassVar[str]
    forms: ClassVar[frozenset[Form]]

    @abstractmethod
    def _unsupported(self, problem: Problem) -> list[str]:
        """Return a description of every element, interaction, or law of a problem the library cannot run.

        Args:
            problem:
                The compiled problem.

        Returns:
            unsupported:
                One description per unsupported item, empty when the problem is supported.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    @abstractmethod
    def _build(self, problem: Problem, rng: np.random.Generator) -> None:
        """Build the library model of a supported problem.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator that seeds the library.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    def _prepare(self, problem: Problem, rng: np.random.Generator, until: float = math.inf) -> None:
        """Check that the library supports a problem and build its model.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator that seeds the library.

            until:
                The end time of the run.

        Raises:
            BackendError:
                The problem has a form, a complex state, an element, an interaction, or a law the library cannot run.
        """
        unsupported = [] if problem.form in self.forms else [f'the {problem.form.value} form']
        unsupported += [f'the complex state {entry.name}' for entry in problem.states.entries if entry.is_complex]
        unsupported += self._unsupported(problem)
        if unsupported:
            items = '; '.join(unsupported)
            error_msg = f'The {self.name} backend cannot run this system. Unsupported: {items}.'
            raise BackendError(error_msg)
        super()._prepare(problem, rng, until)
        self._build(problem, rng)
