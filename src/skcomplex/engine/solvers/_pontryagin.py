"""Solve optimal control problems by Pontryagin's principle, with symbolic costates and a forward-backward sweep."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import math

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, solve

from ...core.laws import BoundaryLaw, EquationLaw, FluxLaw, RateLaw
from ...core.spaces import ContinuousTime
from ...errors import SolverError, ValidationError
from ...types import Array, Cost, GridFunction
from ..problems import OrderKind, SupportSpec
from ._control_solution import ControlSolution
from ._optimal_control import OptimalControlSolver, _ControlProblem

_MIN_RELAXATION = 1e-3
MIN_NODES = 2

_CONTROL_SUPPORT = SupportSpec(
    time={ContinuousTime},
    space={None},
    laws={RateLaw, FluxLaw, EquationLaw, BoundaryLaw},
    orders={OrderKind.INTEGER},
)


class PontryaginSolver(OptimalControlSolver):
    """Optimal control by Pontryagin's principle, with costates and the stationarity condition derived symbolically.

    The solver builds the Hamiltonian `H = L + λ · f` from the running cost `L` and the rates `f` of the states, and
    derives the costate equations `λ' = -∂H/∂x` with `λ(T) = ∂φ/∂x` at the horizon `T` for the terminal cost `φ`. The
    controls solve `∂H/∂u = 0`, projected onto their bounds, or take the bound that minimises `H` where `H` is linear
    in them. A forward-backward sweep integrates the states forward and the costates backward with fourth-order
    Runge-Kutta steps on a grid, and relaxes the controls towards their update until they converge. The weight of the
    update halves whenever the distance of the controls to their update stops falling, which breaks the cycles of the
    sweep. An objective to maximise is minimised with its sign changed. The solution holds the costates.

    Args:
        objective:
            The running cost, a function of the namespace `s` of the states, parameters, and time of the system by
            dotted name, such as `lambda s: s.plant.x**2`, or a number.

        terminal:
            The terminal cost at the horizon, a function of the same namespace or a number.

        sense:
            `'min'` to minimise the objective or `'max'` to maximise it.

        nodes:
            The number of times of the grid, from the start to the horizon.

        relaxation:
            The initial weight of the update of the controls at every sweep, between 0 and 1.

        tolerance:
            The largest relative change of the states, costates, and controls between sweeps at convergence.

        max_iterations:
            The largest number of sweeps.

    Attributes:
        solution_:
            The optimal solution of the run.

    Raises:
        SolverError:
            The solver has no objective, the sweep diverges or does not converge, or the stationarity condition has
            no unique explicit solution.

        ValidationError:
            The sense is unknown, the grid has fewer than 2 nodes, or the relaxation is not in `(0, 1]`.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.elements import Element
        >>> from skcomplex.core.laws import RateLaw
        >>> from skcomplex.specs import ParameterSpec, StateSpec
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.engine.solvers import PontryaginSolver
        >>> class PlantElement(Element):
        ...     def __init__(
        ...         self,
        ...         label,
        ...         x: Annotated[float, StateSpec()] = 1.0,
        ...         u: Annotated[float, ParameterSpec(adaptive=True, bounds=(-10, 10))] = 0.0,
        ...     ):
        ...         super().__init__(label)
        ...         self.x = x
        ...         self.u = u
        ...
        ...     def _get_laws(self, s):
        ...         return [RateLaw(s.x, s.u)]
        >>> solver = PontryaginSolver(objective=lambda s: (s.plant.x**2 + s.plant.u**2) / 2)
        >>> simulation = Simulation(System(elements=[PlantElement('plant')]), solver=solver).check().run(until=1.0)
        >>> float(round(simulation.trajectory_.values('plant.u')[0], 3))
        -0.762
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return _CONTROL_SUPPORT

    def __init__(
        self,
        objective: Cost | None = None,
        *,
        terminal: Cost = 0.0,
        sense: str = 'min',
        nodes: int = 201,
        relaxation: float = 0.5,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        random_state: int | np.random.Generator | None = None,
        until: float = math.inf,
    ) -> None:
        self.objective = objective
        self.terminal = terminal
        self.sense = sense
        self.nodes = nodes
        self.relaxation = relaxation
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.until = until

    def _check(self) -> None:
        """Raise an error for a missing objective, an unknown sense, too few nodes, or a bad relaxation.

        Raises:
            SolverError:
                The solver has no objective.

            ValidationError:
                The sense is unknown, the grid has fewer than 2 nodes, or the relaxation is not in `(0, 1]`.
        """
        super()._check()
        if not (isinstance(self.nodes, int) and self.nodes >= MIN_NODES):
            error_msg = f'nodes = {self.nodes!r} must be an integer of at least {MIN_NODES}.'
            raise ValidationError(error_msg)
        if not 0 < self.relaxation <= 1:
            error_msg = f'relaxation = {self.relaxation!r} must lie in (0, 1].'
            raise ValidationError(error_msg)

    def _costates(self, times: Array, states: Array, controls: Array, p: Array) -> Array:
        """Return the costates on a grid, integrated backward from the horizon by Runge-Kutta steps."""
        costates = np.zeros_like(states)
        costates[-1] = self._final(times[-1], states[-1], controls[-1], p)
        for k in range(times.size - 1, 0, -1):
            h = times[k] - times[k - 1]
            middle_time = times[k] - h / 2
            x_middle, u_middle = (states[k] + states[k - 1]) / 2, (controls[k] + controls[k - 1]) / 2
            lam = costates[k]
            k1 = self._adjoint(times[k], states[k], controls[k], p, lam)
            k2 = self._adjoint(middle_time, x_middle, u_middle, p, lam - h / 2 * k1)
            k3 = self._adjoint(middle_time, x_middle, u_middle, p, lam - h / 2 * k2)
            k4 = self._adjoint(times[k - 1], states[k - 1], controls[k - 1], p, lam - h * k3)
            costates[k - 1] = lam - h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return costates

    def _update(self, times: Array, states: Array, controls: Array, costates: Array, p: Array) -> Array:
        """Return the controls that minimise the Hamiltonian on a grid, within their bounds."""
        lower, upper = self.control_problem_.bounds
        values = self._law(times, states, controls, p, costates)
        if self._bang_bang:
            return np.where(values > 0, lower, upper)
        return np.clip(values, lower, upper)

    def _set_up_control(self, control_problem: _ControlProblem) -> None:
        """Derive the costate equations, the terminal costates, and the control law symbolically.

        Args:
            control_problem:
                The optimal control problem.

        Raises:
            SolverError:
                The stationarity condition has no unique explicit solution in the controls.
        """
        costates = [SR.var(f'_costate_{index}') for index in range(len(control_problem.states))]
        hamiltonian = control_problem.running + sum(
            (lam * rate for lam, rate in zip(costates, control_problem.rates, strict=True)),
            SR(0),
        )
        controls = control_problem.control_symbols
        gradients = [hamiltonian.diff(control) for control in controls]
        self._bang_bang = self._is_linear(gradients, controls)
        law = gradients if self._bang_bang else self._stationary_controls(gradients, controls)
        self._adjoint: GridFunction = control_problem.function(
            [-hamiltonian.diff(item) for item in control_problem.states],
            costates,
        )
        self._final: GridFunction = control_problem.function(
            [control_problem.terminal.diff(item) for item in control_problem.states],
        )
        self._law: GridFunction = control_problem.function(law, costates)

    def _solve(self, x0: Array, p: Array, horizon: float) -> ControlSolution:
        """Return the optimal solution by a forward-backward sweep with relaxed control updates.

        Args:
            x0:
                The initial state vector.

            p:
                The parameter vector.

            horizon:
                The end of the run.

        Returns:
            solution:
                The optimal solution with its costates.

        Raises:
            SolverError:
                The sweep diverges or does not converge within the largest number of sweeps.
        """
        control_problem = self.control_problem_
        times = np.linspace(0.0, horizon, self.nodes)
        lower, upper = control_problem.bounds
        start = np.clip(x0[control_problem.control_indices], lower, upper)
        controls = np.tile(start, (times.size, 1))
        initial = x0[control_problem.indices]
        states = self._integrate(initial, controls, times, p)
        costates = self._costates(times, states, controls, p)
        weight, distance = self.relaxation, np.inf
        for iteration in range(1, self.max_iterations + 1):
            update = self._update(times, states, controls, costates, p)
            gap = float(np.abs(update - controls).sum())
            if gap >= distance:
                weight = max(weight / 2, _MIN_RELAXATION)
            distance = gap
            new_controls = (1 - weight) * controls + weight * update
            new_states = self._integrate(initial, new_controls, times, p)
            new_costates = self._costates(times, new_states, new_controls, p)
            if not (np.all(np.isfinite(new_states)) and np.all(np.isfinite(new_costates))):
                error_msg = (
                    f'The forward-backward sweep diverged at sweep {iteration}, as it does near unstable saddle paths. '
                    'Lower the relaxation, shorten the horizon, or use the CollocationSolver solver.'
                )
                raise SolverError(error_msg)
            pairs = [(new_controls, controls), (new_states, states), (new_costates, costates)]
            controls, states, costates = new_controls, new_states, new_costates
            if self._converged(self.tolerance, pairs):
                named = {name: costates[:, index] for index, name in enumerate(control_problem.names)}
                return self._solution(x0, times, states, controls, p, iterations=iteration, costates=named)
        error_msg = (
            f'The forward-backward sweep did not converge in {self.max_iterations} sweeps. Lower the relaxation, raise '
            'max_iterations, or use the CollocationSolver solver.'
        )
        raise SolverError(error_msg)

    @staticmethod
    def _is_linear(gradients: list[Expression], controls: list[Expression]) -> bool:
        """Check whether no gradient of the Hamiltonian by a control depends on any control."""
        return all(gradient.diff(control).is_trivial_zero() for gradient in gradients for control in controls)

    @staticmethod
    def _stationary_controls(gradients: list[Expression], controls: list[Expression]) -> list[Expression]:
        """Return the controls that make every gradient of the Hamiltonian vanish, solved symbolically."""
        solutions = solve([gradient == 0 for gradient in gradients], *controls, solution_dict=True)
        explicit = [
            solution
            for solution in solutions
            if set(solution) == set(controls)
            and all(not set(SR(value).variables()) & set(controls) for value in solution.values())
        ]
        if len(explicit) != 1:
            error_msg = (
                'The condition that the Hamiltonian is stationary in the controls has no unique explicit solution. Use '
                'the CollocationSolver solver, which needs no such solution.'
            )
            raise SolverError(error_msg)
        return [SR(explicit[0][control]) for control in controls]

    @staticmethod
    def _converged(tolerance: float, pairs: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]) -> bool:
        """Check whether every new array differs from the old one by at most a relative tolerance in the sum norm."""
        return all(
            np.abs(new - old).sum() <= tolerance * max(np.abs(new).sum(), np.finfo(float).tiny) for new, old in pairs
        )
