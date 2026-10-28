"""Define the control problem spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression

from ..errors import SolverError
from ..types import Array, Cost, FormKind, GridFunction
from ._base_spec import BaseSpec
from ._control_spec import _ControlSpec
from ._problem_spec import ProblemSpec


@dataclass(frozen=True)
class _ControlProblemSpec(BaseSpec):
    """Optimal control problem of a compiled system, with its cost written to be minimised."""

    problem: ProblemSpec
    names: tuple[str, ...]
    indices: npt.NDArray[np.intp]
    states: tuple[Expression, ...]
    controls: tuple[_ControlSpec, ...]
    parameters: tuple[Expression, ...]
    rates: tuple[Expression, ...]
    running: Expression
    terminal: Expression
    sign: float
    lower: Array = field(repr=False)

    @property
    def time(self) -> Expression:
        """Time symbol of the expressions."""
        return self.problem.time

    @property
    def control_symbols(self) -> list[Expression]:
        """SimpleNamespace of the controls."""
        return [control.symbol for control in self.controls]

    @property
    def control_indices(self) -> npt.NDArray[np.intp]:
        """Indices of the controls in the state vector."""
        return np.array([control.index for control in self.controls], dtype=np.intp)

    @property
    def bounds(self) -> tuple[Array, Array]:
        """Lower and upper bounds of the controls."""
        return (
            np.array([control.lower for control in self.controls]),
            np.array([control.upper for control in self.controls]),
        )

    def function(self, expressions: Sequence[Any], extra: Sequence[Expression] = ()) -> GridFunction:
        """Return a function `f(t, x, u, p, *extra)` of expressions, whose last axes run over symbols and values."""
        from ..engine.problems import ExpressionPrinter
        arguments = [*self.states, *self.control_symbols, *self.parameters, *extra]
        printer = ExpressionPrinter(time=self.time, compiled=True).check().prepare(arguments)
        compiled = printer.run([SR(item) for item in expressions])
        count = len(expressions)

        def evaluate(t: float | Array, x: Array, u: Array, p: Array, *values: Array) -> npt.NDArray[np.float64]:
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
        from ..engine._utils import symbols_of
        known = {*self.states, *self.control_symbols, *self.parameters, self.time}
        unknown = sorted({str(item) for item in symbols_of([SR(value) for value in expressions]) - known})
        if unknown:
            error_msg = (
                f'The dynamics or the objective read {", ".join(unknown)}, which are not variables of single '
                'elements or interactions. Optimal control solvers take systems of single elements.'
            )
            raise SolverError(error_msg)

    @classmethod
    def of(cls, problem: ProblemSpec, objective: Cost, terminal: Cost, sense: str) -> '_ControlProblemSpec':
        """Return the optimal control problem of a compiled problem with bounded controls and an objective."""
        from ..engine.compile import ObservableEvaluator
        cls._check_supported(problem)
        symbols = cls._single_symbols(problem)
        controls = cls._controls(problem, symbols)
        rates = {problem.equation_symbols[name]: SR(value) for name, value in problem.equations.items()}
        controlled = {control.index for control in controls}
        indices = np.array([index for index in range(len(symbols)) if index not in controlled], dtype=np.intp)
        names = [name for entry in problem.states.entries for name in entry.component_names]
        sign = 1.0 if sense == 'min' else -1.0
        namespace = ObservableEvaluator._namespace_of(problem)
        control_problem = cls(
            problem=problem,
            names=tuple(names[index] for index in indices),
            indices=indices,
            states=tuple(symbols[index] for index in indices),
            controls=controls,
            parameters=tuple(problem.parameter_symbols),
            rates=tuple(rates.get(symbols[index], SR(0)) for index in indices),
            running=sign * cls._cost_value(objective, namespace),
            terminal=sign * cls._cost_value(terminal, namespace),
            sign=sign,
            lower=cls._lower_bounds(problem, indices),
        )
        control_problem.check_symbols([*control_problem.rates, control_problem.running, control_problem.terminal])
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
        from ..engine.solvers._optimal_control import _NONNEGATIVE_DOMAINS
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
