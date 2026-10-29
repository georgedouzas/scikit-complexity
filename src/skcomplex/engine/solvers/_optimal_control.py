"""Read optimal control problems from compiled systems and follow their solutions over the horizon of a run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, SR, Expression, oo
from sage.sets.real_set import RealSet

from ...errors import SolverError, ValidationError
from ...specs import ControlSolutionSpec
from ...specs._control_spec import _ControlSpec
from ...types import Array, FormKind
from .._utils import symbols_of
from ..compile import ObservableEvaluator
from ..problems import ExpressionPrinter, ProblemSpec
from ..problems._control_problem_spec import _ControlProblemSpec
from ..types import Cost, GridFunction
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

    def _prepare(self, context: ProblemSpec | None) -> None:
        """Prepare the solver for a problem and read its optimal control problem.

        Args:
            context:
                The compiled problem.

        Raises:
            SolverError:
                The problem is not an ODE, or has no bounded adaptive parameter or laws the solver cannot
                differentiate symbolically, or the run has no end.
        """
        super()._prepare(context)
        problem = self.problem_
        control_problem = self._control_problem(problem)
        if not math.isfinite(self.until):
            error_msg = 'An optimal control run needs a finite horizon. Run the simulation with `run(until=...)`.'
            raise SolverError(error_msg)
        self.control_problem_ = control_problem
        self.horizon_ = float(self.until)
        self.rates_ = self._function(control_problem, control_problem.rates)
        self.running_ = self._function(control_problem, [control_problem.running])
        self.terminal_ = self._function(control_problem, [control_problem.terminal])
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

    @staticmethod
    def _function(
        control_problem: _ControlProblemSpec,
        expressions: Sequence[Expression],
        extra: Sequence[Expression] = (),
    ) -> GridFunction:
        """Return a function `f(t, x, u, p, *extra)` of expressions, whose last axes run over symbols and values."""
        arguments = [*control_problem.states, *control_problem.control_symbols, *control_problem.parameters, *extra]
        printer = ExpressionPrinter(time=control_problem.time, compiled=True).check().prepare(arguments)
        compiled = printer.run([SR(item) for item in expressions])
        if not callable(compiled):
            error_msg = f'The printer returned a {type(compiled).__name__}, not a compiled function.'
            raise TypeError(error_msg)
        count = len(expressions)

        def evaluate(t: float | Array, x: Array, u: Array, p: Array, *values: Array) -> npt.NDArray[np.float64]:
            if np.ndim(x) == 1 and np.ndim(t) == 0:
                point = compiled(t, *x, *u, *p, *(item for value in values for item in value))
                if not isinstance(point, list):
                    error_msg = f'The compiled function returned a {type(point).__name__}, not a list of values.'
                    raise TypeError(error_msg)
                return np.array(point, dtype=np.float64)
            times = np.asarray(t, dtype=float)
            columns = [*np.moveaxis(x, -1, 0), *np.moveaxis(u, -1, 0), *p]
            columns += [item for value in values for item in np.moveaxis(value, -1, 0)]
            shape = np.broadcast_shapes(times.shape, *(np.shape(item) for item in columns))
            if not count:
                return np.zeros((*shape, 0))
            results = compiled(times, *columns)
            if not isinstance(results, list):
                error_msg = f'The compiled function returned a {type(results).__name__}, not a list of values.'
                raise TypeError(error_msg)
            return np.stack([np.broadcast_to(np.asarray(item, dtype=float), shape) for item in results], axis=-1)

        return evaluate

    @staticmethod
    def _check_symbols(control_problem: _ControlProblemSpec) -> None:
        """Raise an error for a rate or a cost that reads a symbol other than states, controls, parameters, and time."""
        expressions = [*control_problem.rates, control_problem.running, control_problem.terminal]
        known = {
            *control_problem.states,
            *control_problem.control_symbols,
            *control_problem.parameters,
            control_problem.time,
        }
        unknown = sorted({str(item) for item in symbols_of([SR(value) for value in expressions]) - known})
        if unknown:
            error_msg = (
                f'The dynamics or the objective read {", ".join(unknown)}, which are not variables of single '
                'elements or interactions. Optimal control solvers take systems of single elements.'
            )
            raise SolverError(error_msg)

    def _control_problem(self, problem: ProblemSpec) -> _ControlProblemSpec:
        """Return the optimal control problem of a compiled problem, with its bounded controls and the objective."""
        self._check_supported(problem)
        objective = self.objective
        if objective is None:
            error_msg = f'{type(self).__name__} needs an objective. Pass `objective`, a function or a number.'
            raise SolverError(error_msg)
        symbols = self._single_symbols(problem)
        controls = self._controls(problem, symbols)
        rates = {problem.equation_symbols[name]: SR(value) for name, value in problem.equations.items()}
        controlled = {control.index for control in controls}
        indices = np.array([index for index in range(len(symbols)) if index not in controlled], dtype=np.intp)
        names = [name for entry in problem.states.entries for name in entry.component_names]
        sign = 1.0 if self.sense == 'min' else -1.0
        namespace = ObservableEvaluator._namespace_of(problem)
        control_problem = _ControlProblemSpec(
            problem=problem,
            names=tuple(names[index] for index in indices),
            indices=indices,
            states=tuple(symbols[index] for index in indices),
            controls=controls,
            parameters=tuple(problem.parameter_symbols),
            rates=tuple(rates.get(symbols[index], SR(0)) for index in indices),
            running=sign * self._cost_value(objective, namespace),
            terminal=sign * self._cost_value(self.terminal, namespace),
            sign=sign,
            lower=self._lower_bounds(problem, indices),
        )
        self._check_symbols(control_problem)
        return control_problem

    @staticmethod
    def _cost_value(cost: Cost, namespace: SimpleNamespace) -> Expression:
        """Return the expression of a cost, a function of the namespace of the system or a number."""
        return SR(cost(namespace) if callable(cost) else cost)

    @staticmethod
    def _single_symbols(problem: ProblemSpec) -> list[Expression]:
        """Return the state symbols in state vector order, after checking that every state is a real element scalar."""
        if any(element.members is not None for element in problem.elements):
            error_msg = 'Optimal control solvers take systems of single elements, but the system has groups.'
            raise SolverError(error_msg)
        symbols = problem.state_symbols
        if len(symbols) != problem.states.size:
            error_msg = 'Optimal control solvers take real states, but the system has complex states.'
            raise SolverError(error_msg)
        return symbols

    @staticmethod
    def _check_supported(problem: ProblemSpec) -> None:
        """Raise an error for a problem with laws that have no symbolic vector field or no finite state space."""
        if problem.form != FormKind.ODE:
            error_msg = (
                f'Optimal control solvers take ordinary differential equations, but the system is a {problem.form}.'
            )
            raise SolverError(error_msg)
        if problem.has_callables or problem.has_group_sums or problem.delays or problem.events:
            error_msg = (
                'Optimal control solvers derive the costates and the stationarity condition symbolically, so they take '
                'symbolic rate laws without callables, sums over members, delays, or events.'
            )
            raise SolverError(error_msg)

    @staticmethod
    def _lower_bounds(problem: ProblemSpec, indices: Sequence[int] | npt.NDArray[np.intp]) -> Array:
        """Return the lower bound of the states at indices: 0 for nonnegative domains and minus infinity for others."""
        lower = np.full(problem.states.size, -np.inf)
        for entry in problem.states.entries:
            if entry.domain in _NONNEGATIVE_DOMAINS:
                lower[entry.indices] = 0.0
        return lower[list(indices)]

    @staticmethod
    def _controls(problem: ProblemSpec, symbols: Sequence[Expression]) -> tuple[_ControlSpec, ...]:
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
            controls.append(_ControlSpec(name, entry.start, symbols[entry.start], lower, upper))
        if not controls:
            error_msg = (
                'The system has no bounded adaptive parameter to control. Declare a control input with '
                '`Annotated[float, ParameterSpec(adaptive=True, bounds=(lower, upper))]`.'
            )
            raise SolverError(error_msg)
        return tuple(controls)

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
