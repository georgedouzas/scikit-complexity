"""Read optimal control problems from compiled systems and follow their solutions over the horizon of a run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from abc import abstractmethod
from collections.abc import Mapping
from typing import cast

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ...errors import SolverError, ValidationError
from ...specs import ControlSolutionSpec, ProblemSpec
from ...specs._control_problem_spec import _ControlProblemSpec
from ...types import Array, Cost, FormKind
from ._solver import Solver

TIME_TOLERANCE = 1e-9
_SENSES = ('min', 'max')
_NONNEGATIVE_DOMAINS = frozenset({RealSet.open(0, oo), RealSet.closed_open(0, oo), NN})






class OptimalControlSolver(Solver):
    """Base class of solvers that compute the controls minimising or maximising an objective over a run of a system.

    The solver reads the bounded adaptive parameters of the system as its controls. Its objective is the running cost
    `objective` integrated over the run plus the terminal cost `terminal` at the horizon, the time `until` of the run.
    Each cost is a function of the namespace `s` of the states, parameters, and time of the system by dotted name, such
    as `lambda s: s.plant.x**2 + s.plant.u**2`, or a number. The solver minimises the objective for the sense `'min'`
    and maximises it for `'max'`. It solves the problem over the whole run from the initial state and the initial
    parameters at the start of the run, and then follows the solution frame by frame. Subclasses take `objective`,
    `terminal`, and `sense` as arguments, implement `_solve`, and may implement `_prepare`.

    Attributes:
        control_problem_:
            The optimal control problem of the compiled system.

        horizon_:
            The end of the run, the horizon of the objective.

        solution_:
            The optimal solution of the run.
    """

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.ODE
    objective: Cost | None
    terminal: Cost
    sense: str

    def _check(self) -> None:
        """Raise an error for a missing objective or an unknown sense.

        Raises:
            SolverError:
                The solver has no objective.

            ValidationError:
                The sense is neither `'min'` nor `'max'`.
        """
        if self.objective is None:
            error_msg = (
                f'{type(self).__name__} has no objective. Pass the running cost as `objective`, a function of the '
                'namespace of the system such as `lambda s: s.plant.x**2`.'
            )
            raise SolverError(error_msg)
        if self.sense not in _SENSES:
            error_msg = f'Unknown sense {self.sense!r} of the objective. Use one of {", ".join(_SENSES)}.'
            raise ValidationError(error_msg)

    def _integrate(self, x0: Array, controls: Array, times: Array, p: Array) -> Array:
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

    def _cost(self, times: Array, states: Array, controls: Array, p: Array) -> float:
        """Return the value of the objective, in its own sense, for states and controls on a grid of times."""
        running = self.running_(times, states, controls, p)[:, 0]
        terminal = self.terminal_(times[-1], states[-1], controls[-1], p)[0]
        return float(self.control_problem_.sign * (np.trapezoid(running, times) + terminal))

    def _solution(
        self,
        x0: Array,
        times: Array,
        states: Array,
        controls: Array,
        p: Array,
        *,
        iterations: int,
        costates: Mapping[str, Array] | None = None,
    ) -> ControlSolutionSpec:
        """Return the solution of steered states and controls on a grid, with its cost."""
        control_problem = self.control_problem_
        full = np.tile(x0, (times.size, 1))
        full[:, control_problem.indices] = states
        full[:, control_problem.control_indices] = controls
        cost = self._cost(times, states, controls, p)
        return ControlSolutionSpec(times, full, dict(costates or {}), cost, iterations)

    def _prepare(self, problem: ProblemSpec) -> None:
        """Prepare the solver for a problem and read its optimal control problem.

        Args:
            problem:
                The compiled problem.

        Raises:
            SolverError:
                The problem is not an ODE, or has no bounded adaptive parameter or laws the solver cannot
                differentiate symbolically, or the run has no end.
        """
        super()._prepare(problem)
        control_problem = _ControlProblemSpec.of(problem, cast('Cost', self.objective), self.terminal, self.sense)
        if not math.isfinite(self.until):
            error_msg = 'An optimal control run needs a finite horizon. Run the simulation with `run(until=...)`.'
            raise SolverError(error_msg)
        self.control_problem_ = control_problem
        self.horizon_ = float(self.until)
        self.rates_ = control_problem.function(control_problem.rates)
        self.running_ = control_problem.function([control_problem.running])
        self.terminal_ = control_problem.function([control_problem.terminal])
        self._set_up_control(control_problem)
        self.solution_ = self._solve(self.initial_state_.copy(), problem.initial_parameters.copy(), self.horizon_)
        self.initial_state_ = self.solution_.state_at(0.0)

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
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

    def _set_up_control(self, control_problem: _ControlProblemSpec) -> None:
        """Derive what the solver needs from the optimal control problem, once per compiled problem.

        Args:
            control_problem:
                The optimal control problem.
        """
        del control_problem

    @abstractmethod
    def _solve(self, x0: Array, p: Array, horizon: float) -> ControlSolutionSpec:
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
