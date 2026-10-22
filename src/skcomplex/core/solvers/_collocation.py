"""Solve optimal control problems by direct collocation, a transcription solved by nonlinear programming."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, minimize

from ..base import ComponentKind, DomainKind, LawKind, SolverError, SpaceKind, Supports, TimeKind, ValidationError
from ..compile import Vector
from ._base import register_solver
from ._optimal import ControlSolution, GridFunction, OptimalControlSolver, _ControlProblem

SCHEMES = ('trapezoidal', 'hermite-simpson')
METHODS = ('SLSQP', 'trust-constr')
MIN_NODES = 2

type Matrix = npt.NDArray[np.float64]


def _trapezoidal(times: Vector) -> tuple[Vector, Matrix, Matrix, Vector]:
    """Return the points, the defect coefficients of states and rates, and the quadrature weights of trapezoids."""
    steps = np.diff(times)
    rows = np.arange(steps.size)
    states = np.zeros((steps.size, times.size))
    rates = np.zeros((steps.size, times.size))
    states[rows, rows], states[rows, rows + 1] = -1.0, 1.0
    rates[rows, rows], rates[rows, rows + 1] = -steps / 2, -steps / 2
    weights = np.zeros(times.size)
    np.add.at(weights, rows, steps / 2)
    np.add.at(weights, rows + 1, steps / 2)
    return times, states, rates, weights


def _hermite_simpson(times: Vector) -> tuple[Vector, Matrix, Matrix, Vector]:
    """Return the points with midpoints, the defect coefficients of states and rates, and Simpson's weights."""
    steps = np.diff(times)
    points = np.empty(2 * times.size - 1)
    points[0::2], points[1::2] = times, times[:-1] + steps / 2
    intervals = np.arange(steps.size)
    first, middle, last = 2 * intervals, 2 * intervals + 1, 2 * intervals + 2
    interpolation, quadrature = 2 * intervals, 2 * intervals + 1
    states = np.zeros((2 * steps.size, points.size))
    rates = np.zeros((2 * steps.size, points.size))
    states[interpolation, middle], states[interpolation, first], states[interpolation, last] = 1.0, -0.5, -0.5
    rates[interpolation, first], rates[interpolation, last] = -steps / 8, steps / 8
    states[quadrature, last], states[quadrature, first] = 1.0, -1.0
    rates[quadrature, first], rates[quadrature, middle], rates[quadrature, last] = (
        -steps / 6,
        -2 * steps / 3,
        -steps / 6,
    )
    weights = np.zeros(points.size)
    np.add.at(weights, first, steps / 6)
    np.add.at(weights, middle, 2 * steps / 3)
    np.add.at(weights, last, steps / 6)
    return points, states, rates, weights


class Collocation(OptimalControlSolver):
    """Optimal control by direct collocation, a transcription of the problem solved by nonlinear programming.

    The states and the controls at the points of a grid are the variables of a nonlinear programme. The trapezoidal
    scheme requires the states of neighbouring nodes to differ by the trapezoidal integral of their rates, and the
    Hermite-Simpson scheme adds a midpoint to every interval, with a cubic interpolation of the states and Simpson's
    integral. The cost is the quadrature of the running cost over the points plus the terminal cost. The controls lie
    within their bounds and the states of nonnegative domains stay nonnegative. SciPy's `minimize` solves the programme
    with the gradient of the cost and the Jacobian of the constraints derived symbolically. An objective to maximise is
    minimised with its sign changed.

    Args:
        nodes:
            The number of times of the grid, from the start to the horizon.

        scheme:
            `'trapezoidal'` or `'hermite-simpson'`.

        method:
            The method of `scipy.optimize.minimize`, `'SLSQP'` or `'trust-constr'`.

        tolerance:
            The tolerance of the optimiser.

        max_iterations:
            The largest number of iterations of the optimiser.

    Attributes:
        solution_:
            The optimal solution of the run.

    Raises:
        ValidationError:
            The scheme or the method is unknown, or the grid has fewer than 2 nodes.

        SolverError:
            The optimiser does not converge.

    Examples:
        >>> from skcomplex.core.base import parameter
        >>> from skcomplex.core.components import Element
        >>> from skcomplex.core.laws import Objective, Rate
        >>> from skcomplex.core.runtime import Simulation
        >>> from skcomplex.core.solvers import Collocation
        >>> from skcomplex.core.system import System
        >>> Plant = Element.define(
        ...     'Plant',
        ...     states={'x': 1.0},
        ...     parameters={'u': parameter(0.0, adaptive=True, bounds=(-10, 10))},
        ...     laws=lambda s: [Rate(s.x, s.u), Objective((s.x**2 + s.u**2) / 2)],
        ... )
        >>> solver = Collocation(nodes=51, scheme='hermite-simpson')
        >>> simulation = Simulation(System(components=[Plant('plant')]), solver=solver).run(until=1.0)
        >>> round(simulation.trajectory_['plant.u'][0], 3)
        -0.762
    """

    supports: ClassVar[Supports] = Supports(
        time={TimeKind.CONTINUOUS},
        space={SpaceKind.NONE},
        laws={LawKind.RATE, LawKind.EQUATION, LawKind.OBJECTIVE},
        domains={DomainKind.REAL, DomainKind.ANGLE},
        components={ComponentKind.ELEMENT, ComponentKind.SYSTEM},
    )

    def __init__(
        self,
        nodes: int = 101,
        scheme: str = 'hermite-simpson',
        method: str = 'SLSQP',
        tolerance: float = 1e-10,
        max_iterations: int = 1000,
    ) -> None:
        self.nodes = nodes
        self.scheme = scheme
        self.method = method
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def _check_settings(self) -> None:
        """Raise an error for an unknown scheme or method, or a grid of fewer than two nodes."""
        if self.scheme not in SCHEMES:
            error_msg = f'Unknown collocation scheme {self.scheme!r}. Use one of {", ".join(SCHEMES)}.'
            raise ValidationError(error_msg)
        if self.method not in METHODS:
            error_msg = f'Unknown optimiser method {self.method!r}. Use one of {", ".join(METHODS)}.'
            raise ValidationError(error_msg)
        if not (isinstance(self.nodes, int) and self.nodes >= MIN_NODES):
            error_msg = f'nodes = {self.nodes!r} must be an integer of at least {MIN_NODES}.'
            raise ValidationError(error_msg)

    def _unpack(self, z: Vector, points: int) -> tuple[Vector, Vector]:
        """Return the states and the controls of the points from the vector of variables."""
        n, m = len(self.control_problem_.states), len(self.control_problem_.controls)
        return z[: points * n].reshape(points, n), z[points * n :].reshape(points, m)

    def _cost_and_gradient(self, z: Vector, points: Vector, weights: Vector, p: Vector) -> tuple[float, Vector]:
        """Return the transcribed cost and its gradient by the variables."""
        x, u = self._unpack(z, points.size)
        values = self.running_(points, x, u, p)[:, 0]
        cost = float(weights @ values + self.terminal_(points[-1], x[-1], u[-1], p)[0])
        gradient_x = weights[:, None] * self._running_states(points, x, u, p)
        gradient_x[-1] += self._terminal_states(points[-1], x[-1], u[-1], p)
        gradient_u = weights[:, None] * self._running_controls(points, x, u, p)
        return cost, np.concatenate([gradient_x.ravel(), gradient_u.ravel()])

    def _defects(self, z: Vector, points: Vector, coefficients: tuple[Matrix, Matrix], x0: Vector, p: Vector) -> Vector:
        """Return the defects of the collocation conditions and of the initial state."""
        x, u = self._unpack(z, points.size)
        states, rates = coefficients
        defects = states @ x + rates @ self.rates_(points, x, u, p)
        return np.concatenate([defects.T.ravel(), x[0] - x0])

    def _jacobian(self, z: Vector, points: Vector, coefficients: tuple[Matrix, Matrix], p: Vector) -> Matrix:
        """Return the Jacobian of the defects by the variables, from the symbolic Jacobians of the rates."""
        x, u = self._unpack(z, points.size)
        states, rates = coefficients
        n, m, size = x.shape[1], u.shape[1], points.size
        by_states = self._rates_states(points, x, u, p).reshape(size, n, n)
        by_controls = self._rates_controls(points, x, u, p).reshape(size, n, m)
        rows = states.shape[0]
        jacobian = np.zeros((rows * n + n, size * (n + m)))
        for i in range(n):
            block = slice(i * rows, (i + 1) * rows)
            for j in range(n):
                jacobian[block, j : size * n : n] = rates * by_states[:, i, j] + (states if i == j else 0.0)
            for j in range(m):
                jacobian[block, size * n + j :: m] = rates * by_controls[:, i, j]
            jacobian[rows * n + i, i] = 1.0
        return jacobian

    def _bounds(self, points: int) -> Bounds:
        """Return the bounds of the variables: the lower bounds of the states and the bounds of the controls."""
        lower, upper = self.control_problem_.bounds
        state_lower = np.tile(self.control_problem_.lower, points)
        state_upper = np.full(state_lower.size, np.inf)
        return Bounds(
            np.concatenate([state_lower, np.tile(lower, points)]),
            np.concatenate([state_upper, np.tile(upper, points)]),
        )

    def prepare(self, control_problem: _ControlProblem) -> None:
        """Derive the Jacobians of the rates and the gradients of the costs symbolically.

        Args:
            control_problem:
                The optimal control problem.

        Raises:
            ValidationError:
                The scheme or the method is unknown, or the grid has fewer than 2 nodes.
        """
        self._check_settings()
        states, controls = control_problem.states, control_problem.control_symbols
        rates, running = control_problem.rates, control_problem.running
        self._rates_states: GridFunction = control_problem.function(
            [rate.diff(item) for rate in rates for item in states],
        )
        self._rates_controls: GridFunction = control_problem.function(
            [rate.diff(item) for rate in rates for item in controls],
        )
        self._running_states: GridFunction = control_problem.function([running.diff(item) for item in states])
        self._running_controls: GridFunction = control_problem.function([running.diff(item) for item in controls])
        self._terminal_states: GridFunction = control_problem.function(
            [control_problem.terminal.diff(item) for item in states],
        )

    def solve(self, x0: Vector, p: Vector, horizon: float) -> ControlSolution:
        """Return the optimal solution of the transcribed nonlinear programme.

        Args:
            x0:
                The initial state vector.

            p:
                The parameter vector.

            horizon:
                The end of the run.

        Returns:
            solution:
                The optimal solution at the points of the grid.

        Raises:
            SolverError:
                The optimiser does not converge.
        """
        control_problem = self.control_problem_
        transcribe = _trapezoidal if self.scheme == 'trapezoidal' else _hermite_simpson
        points, states, rates, weights = transcribe(np.linspace(0.0, horizon, self.nodes))
        lower, upper = control_problem.bounds
        initial = x0[control_problem.indices]
        guess_controls = np.tile(np.clip(x0[control_problem.control_indices], lower, upper), (points.size, 1))
        with np.errstate(all='ignore'):
            guess_states = np.maximum(self._integrate(initial, guess_controls, points, p), control_problem.lower)
        if not np.all(np.isfinite(guess_states)):
            guess_states = np.tile(initial, (points.size, 1))
        coefficients = (states, rates)
        constraint: dict[str, Any] = {
            'type': 'eq',
            'fun': lambda z: self._defects(z, points, coefficients, initial, p),
            'jac': lambda z: self._jacobian(z, points, coefficients, p),
        }
        result = minimize(
            lambda z: self._cost_and_gradient(z, points, weights, p),
            np.concatenate([guess_states.ravel(), guess_controls.ravel()]),
            jac=True,
            bounds=self._bounds(points.size),
            constraints=[constraint],
            method=self.method,
            tol=self.tolerance,
            options={'maxiter': self.max_iterations},
        )
        if not result.success:
            error_msg = (
                f'The collocation programme did not converge: {result.message} Raise max_iterations, change the '
                'number of nodes, or try the other scheme or method.'
            )
            raise SolverError(error_msg)
        x, u = self._unpack(result.x, points.size)
        return self._solution(x0, points, x, u, p, iterations=int(result.nit))


register_solver('collocation', Collocation)
