"""Read optimal control problems from compiled systems and follow their solutions over the horizon of a run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR

from ..base import Domain, Expression, Form, SolverError, symbols_of
from ..compile import Problem, Vector, numpy_function
from ._base import Stepper

TIME_TOLERANCE = 1e-9
NONNEGATIVE_DOMAINS = frozenset({Domain.POSITIVE, Domain.NONNEGATIVE, Domain.NONNEGATIVE_INTEGER})

type GridFunction = Callable[..., npt.NDArray[np.float64]]


def _single_symbols(problem: Problem) -> list[Expression]:
    """Return the state symbols in state vector order, after checking that every state is a real scalar of elements."""
    if any(element.members is not None for element in problem.elements):
        error_msg = 'Optimal control solvers take systems of single elements, but the system has groups.'
        raise SolverError(error_msg)
    symbols = problem.state_symbols
    if len(symbols) != problem.states.size:
        error_msg = 'Optimal control solvers take real states, but the system has complex states.'
        raise SolverError(error_msg)
    return symbols


def _check_supported(problem: Problem) -> None:
    """Raise an error for a problem with laws that have no symbolic vector field or no finite state space."""
    if problem.form is not Form.ODE:
        error_msg = f'Optimal control solvers take ordinary differential equations, but the system is a {problem.form}.'
        raise SolverError(error_msg)
    if problem.has_callables or problem.has_group_sums or problem.delays or problem.events:
        error_msg = (
            'Optimal control solvers derive the costates and the stationarity condition symbolically, so they take '
            'symbolic rate laws without callables, sums over members, delays, or events.'
        )
        raise SolverError(error_msg)
    if len(problem.objectives) != 1:
        error_msg = (
            f'Optimal control solvers optimise one objective law, but the system has {len(problem.objectives)}. Give '
            'the system one Objective law.'
        )
        raise SolverError(error_msg)


def _lower_bounds(problem: Problem, indices: Sequence[int]) -> Vector:
    """Return the lower bound of the states at indices: 0 for nonnegative domains and minus infinity for others."""
    lower = np.full(problem.states.size, -np.inf)
    for entry in problem.states:
        if entry.domain in NONNEGATIVE_DOMAINS:
            lower[entry.indices] = 0.0
    return lower[list(indices)]


@dataclass(frozen=True)
class _Control:
    """Control of an optimal control problem: its dotted name, state index, symbol, and bounds."""

    name: str
    index: int
    symbol: Expression
    lower: float
    upper: float


def _controls(problem: Problem, symbols: Sequence[Expression]) -> tuple[_Control, ...]:
    """Return the bounded adaptive parameters of a problem as its controls."""
    controls = []
    for name, declaration in problem.adaptive_parameters.items():
        if declaration.bounds is None:
            continue
        entry = problem.states.entry(name)
        if entry.size != 1:
            error_msg = f'The control {name} is not a scalar. Declare every control as a scalar adaptive parameter.'
            raise SolverError(error_msg)
        lower, upper = declaration.bounds
        controls.append(_Control(name, entry.start, symbols[entry.start], lower, upper))
    if not controls:
        error_msg = (
            'The system has no bounded adaptive parameter to control. Declare a control input with '
            '`parameter(initial, adaptive=True, bounds=(lower, upper))`.'
        )
        raise SolverError(error_msg)
    return tuple(controls)


@dataclass(frozen=True)
class _ControlProblem:
    """Optimal control problem of a compiled system, with its cost written to be minimised."""

    problem: Problem
    names: tuple[str, ...]
    indices: npt.NDArray[np.intp]
    states: tuple[Expression, ...]
    controls: tuple[_Control, ...]
    parameters: tuple[Expression, ...]
    rates: tuple[Expression, ...]
    running: Expression
    terminal: Expression
    sign: float
    lower: Vector = field(repr=False)

    @property
    def time(self) -> Expression:
        """Time symbol of the expressions."""
        return self.problem.time

    @property
    def control_symbols(self) -> list[Expression]:
        """Symbols of the controls."""
        return [control.symbol for control in self.controls]

    @property
    def control_indices(self) -> npt.NDArray[np.intp]:
        """Indices of the controls in the state vector."""
        return np.array([control.index for control in self.controls], dtype=np.intp)

    @property
    def bounds(self) -> tuple[Vector, Vector]:
        """Lower and upper bounds of the controls."""
        return (
            np.array([control.lower for control in self.controls]),
            np.array([control.upper for control in self.controls]),
        )

    def function(self, expressions: Sequence[Any], extra: Sequence[Expression] = ()) -> GridFunction:
        """Return a function `f(t, x, u, p, *extra)` of expressions, whose last axes run over symbols and values."""
        arguments = [*self.states, *self.control_symbols, *self.parameters, *extra]
        compiled = numpy_function(self.time, arguments, [SR(item) for item in expressions])
        count = len(expressions)

        def evaluate(t: float | Vector, x: Vector, u: Vector, p: Vector, *values: Vector) -> npt.NDArray[np.float64]:
            if np.ndim(x) == 1 and np.ndim(t) == 0:
                return np.array(compiled(t, *x, *u, *p, *(item for value in values for item in value)), dtype=float)
            times = np.asarray(t, dtype=float)
            columns = [*np.moveaxis(x, -1, 0), *np.moveaxis(u, -1, 0), *p]
            columns += [item for value in values for item in np.moveaxis(value, -1, 0)]
            shape = np.broadcast_shapes(times.shape, *(np.shape(item) for item in columns))
            if not count:
                return np.zeros((*shape, 0))
            results = compiled(times, *columns)
            return np.stack([np.broadcast_to(np.asarray(item, dtype=float), shape) for item in results], axis=-1)

        return evaluate

    def check_symbols(self, expressions: Sequence[Any]) -> None:
        """Raise an error for an expression that reads a symbol other than states, controls, parameters, and time."""
        known = {*self.states, *self.control_symbols, *self.parameters, self.time}
        unknown = sorted({str(item) for item in symbols_of([SR(value) for value in expressions]) - known})
        if unknown:
            error_msg = (
                f'The dynamics or the objective read {", ".join(unknown)}, which are not variables of single '
                'elements or interactions. Optimal control solvers take systems of single elements.'
            )
            raise SolverError(error_msg)

    @classmethod
    def of(cls, problem: Problem) -> '_ControlProblem':
        """Return the optimal control problem of a compiled problem with one objective and bounded controls."""
        _check_supported(problem)
        symbols = _single_symbols(problem)
        controls = _controls(problem, symbols)
        rates = {problem.equation_symbols[name]: SR(value) for name, value in problem.equations.items()}
        controlled = {control.index for control in controls}
        indices = np.array([index for index in range(len(symbols)) if index not in controlled], dtype=np.intp)
        names = [name for entry in problem.states for name in entry.component_names]
        objective = problem.objectives[0]
        sign = 1.0 if objective.sense == 'min' else -1.0
        control_problem = cls(
            problem=problem,
            names=tuple(names[index] for index in indices),
            indices=indices,
            states=tuple(symbols[index] for index in indices),
            controls=controls,
            parameters=tuple(problem.parameter_symbols),
            rates=tuple(rates.get(symbols[index], SR(0)) for index in indices),
            running=sign * SR(objective.expression),
            terminal=sign * SR(objective.terminal),
            sign=sign,
            lower=_lower_bounds(problem, indices),
        )
        control_problem.check_symbols([*control_problem.rates, control_problem.running, control_problem.terminal])
        return control_problem


@dataclass(frozen=True)
class ControlSolution:
    """Optimal control of a run over its horizon, sampled on a grid of times.

    Args:
        times:
            The times of the grid, from the start to the horizon.

        states:
            The state vector at every time of the grid, with the controls at their optimal values, one row per time.

        costates:
            The costate of every state the controls steer, by state name, one value per time, or empty when the
            solver does not compute costates.

        cost:
            The value of the objective, the running value integrated over the run plus the terminal value.

        iterations:
            The number of sweeps or optimiser iterations the solver took.
    """

    times: Vector
    states: Vector
    costates: Mapping[str, Vector]
    cost: float
    iterations: int

    def state_at(self, t: float) -> Vector:
        """Return the state vector at a time of the run, interpolated linearly between the times of the grid.

        Args:
            t:
                The time.

        Returns:
            state:
                The state vector.
        """
        return np.array([np.interp(t, self.times, column) for column in self.states.T])


class OptimalControlSolver(Stepper):
    """Base class of solvers that compute the controls minimising or maximising the objective of a system.

    The solver reads the bounded adaptive parameters of the system as its controls and its objective law as its cost.
    It solves the problem over the whole run from the initial state and the initial parameters at the start of the
    run, and then follows the solution frame by frame. Subclasses implement `solve`, and may implement `prepare`.

    Attributes:
        control_problem_:
            The optimal control problem of the compiled system.

        horizon_:
            The end of the run, the horizon of the objective.

        solution_:
            The optimal solution of the run.
    """

    form: ClassVar[Form] = Form.ODE

    def _integrate(self, x0: Vector, controls: Vector, times: Vector, p: Vector) -> Vector:
        """Return the steered states on a grid from their initial values, by Runge-Kutta steps under given controls."""
        control_problem = self.control_problem_
        states = np.zeros((times.size, x0.size))
        states[0] = x0
        full = self.problem_.initial_state.copy()
        for k in range(times.size - 1):
            h = times[k + 1] - times[k]
            start, end = controls[k], controls[k + 1]
            middle = (start + end) / 2
            x = states[k]
            k1 = self.rates_(times[k], x, start, p)
            k2 = self.rates_(times[k] + h / 2, x + h / 2 * k1, middle, p)
            k3 = self.rates_(times[k] + h / 2, x + h / 2 * k2, middle, p)
            k4 = self.rates_(times[k + 1], x + h * k3, end, p)
            states[k + 1] = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if self.problem_.boundary is not None:
                full[control_problem.indices] = states[k + 1]
                full[control_problem.control_indices] = end
                states[k + 1] = self.problem_.boundary(times[k + 1], full, p)[control_problem.indices]
        return states

    def _cost(self, times: Vector, states: Vector, controls: Vector, p: Vector) -> float:
        """Return the value of the objective, in its own sense, for states and controls on a grid of times."""
        running = self.running_(times, states, controls, p)[:, 0]
        terminal = self.terminal_(times[-1], states[-1], controls[-1], p)[0]
        return float(self.control_problem_.sign * (np.trapezoid(running, times) + terminal))

    def _solution(
        self,
        x0: Vector,
        times: Vector,
        states: Vector,
        controls: Vector,
        p: Vector,
        *,
        iterations: int,
        costates: Mapping[str, Vector] | None = None,
    ) -> ControlSolution:
        """Return the solution of steered states and controls on a grid, with its cost."""
        control_problem = self.control_problem_
        full = np.tile(x0, (times.size, 1))
        full[:, control_problem.indices] = states
        full[:, control_problem.control_indices] = controls
        cost = self._cost(times, states, controls, p)
        return ControlSolution(times, full, dict(costates or {}), cost, iterations)

    def initialize(self, problem: Problem, rng: np.random.Generator) -> None:
        """Prepare the solver for a problem and read its optimal control problem.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator, which the solver does not use.

        Raises:
            SolverError:
                The problem is not an ODE, or has no objective, no bounded adaptive parameter, or laws the solver
                cannot differentiate symbolically.
        """
        super().initialize(problem, rng)
        control_problem = _ControlProblem.of(problem)
        self.control_problem_ = control_problem
        self.horizon_ = math.nan
        self.rates_ = control_problem.function(control_problem.rates)
        self.running_ = control_problem.function([control_problem.running])
        self.terminal_ = control_problem.function([control_problem.terminal])
        self.prepare(control_problem)

    def plan(self, until: float) -> None:
        """Take the end of the run as the horizon of the objective.

        Args:
            until:
                The end time of the run.

        Raises:
            SolverError:
                The run has no end.
        """
        if not math.isfinite(until):
            error_msg = 'An optimal control run needs a finite horizon. Run the simulation with `run(until=...)`.'
            raise SolverError(error_msg)
        self.horizon_ = float(until)

    def initial_state(self, x: Vector) -> Vector:
        """Solve the optimal control problem from an initial state and return the state of the first frame.

        Args:
            x:
                The initial state vector of the problem.

        Returns:
            state:
                The initial state with the controls at their optimal values.
        """
        self.solution_ = self.solve(x.copy(), self.problem_.initial_parameters.copy(), self.horizon_)
        return self.solution_.state_at(0.0)

    def advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Return the state of the optimal solution at a time of the run.

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
                The optimal state vector at time `until`.

        Raises:
            SolverError:
                The time lies after the horizon the problem was solved for.
        """
        del t, x, p
        if until > self.horizon_ + TIME_TOLERANCE:
            error_msg = (
                f'The optimal control was solved until the horizon {self.horizon_}, so the run cannot go on to {until}.'
                ' Run a new simulation with a later horizon.'
            )
            raise SolverError(error_msg)
        return self.solution_.state_at(until)

    def prepare(self, control_problem: _ControlProblem) -> None:
        """Derive what the solver needs from the optimal control problem, once per compiled problem.

        Args:
            control_problem:
                The optimal control problem.
        """
        del control_problem

    @abstractmethod
    def solve(self, x0: Vector, p: Vector, horizon: float) -> ControlSolution:
        """Return the optimal solution from an initial state over a horizon.

        Args:
            x0:
                The initial state vector.

            p:
                The parameter vector.

            horizon:
                The end of the run.

        Returns:
            solution:
                The optimal solution.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError
