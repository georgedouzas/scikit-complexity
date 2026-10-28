"""Linearise a system into state-space matrices, with its controllability, observability, and optimal gain."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression
from scipy.linalg import solve_continuous_are

from ..core.errors import AnalysisError, ValidationError
from ..engine.problems import Form, Problem
from ._base import BaseAnalysis, Source
from ._kinds import DataKind
from ._stability import StabilityAnalysis

type Matrix = npt.NDArray[np.float64]


class LinearizationAnalysis(BaseAnalysis):
    """Linear state-space model `x' = A x + B u`, `y = C x + D u` of a system near an operating point.

    The matrices are the Jacobians of the rates and of the outputs, derived symbolically and evaluated at the
    operating point. The controls are adaptive parameters, named by their dotted names, and the outputs are states or
    controls. The system is controllable when the rank of `[B, AB, ..., A^(n-1) B]` equals the number of states, and
    observable when the rank of `[C; CA; ...; C A^(n-1)]` does. With weights of the states and the controls, the gain
    `K` of the linear-quadratic regulator `u = -K x` minimises `∫ (x' Q x + u' R u) dt` over an infinite horizon, from
    the continuous algebraic Riccati equation that SciPy solves.

    Args:
        controls:
            The dotted names of the adaptive parameters that are inputs, or `None` for the bounded ones.

        outputs:
            The dotted names of the states and controls that are outputs, or `None` for every state.

        at:
            The values of states and controls at the operating point, by dotted name, or `None` for the initial values.

        state_weights:
            The weight `Q` of the states in the cost of the regulator, a symmetric positive semidefinite matrix, or
            `None` for no regulator.

        control_weights:
            The weight `R` of the controls in the cost of the regulator, a symmetric positive definite matrix, or
            `None` for no regulator.

    Attributes:
        problem_:
            The compiled problem.

        A_:
            The state matrix, the Jacobian of the rates by the states.

        B_:
            The input matrix, the Jacobian of the rates by the controls.

        C_:
            The output matrix, the Jacobian of the outputs by the states.

        D_:
            The feedthrough matrix, the Jacobian of the outputs by the controls.

        states_:
            The names of the state components, in the order of the rows of `A_`.

        controls_:
            The names of the controls, in the order of the columns of `B_`.

        outputs_:
            The names of the outputs, in the order of the rows of `C_`.

        controllability_:
            The rank of the controllability matrix.

        observability_:
            The rank of the observability matrix.

        gain_:
            The gain of the regulator, one row per control and one column per state, or `None` without weights.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.analysis import LinearizationAnalysis
        >>> from skcomplex.core.elements import Element
        >>> from skcomplex.core.laws import RateLaw
        >>> from skcomplex.core.specs import ParameterSpec, StateSpec
        >>> from skcomplex.core.system import System
        >>> class CartElement(Element):
        ...     def __init__(
        ...         self,
        ...         label,
        ...         x: Annotated[float, StateSpec()] = 0.0,
        ...         v: Annotated[float, StateSpec()] = 0.0,
        ...         force: Annotated[float, ParameterSpec(adaptive=True, bounds=(-1, 1))] = 0.0,
        ...     ):
        ...         super().__init__(label)
        ...         self.x = x
        ...         self.v = v
        ...         self.force = force
        ...
        ...     def _get_laws(self, namespace):
        ...         return [RateLaw(namespace.x, namespace.v), RateLaw(namespace.v, namespace.force)]
        >>> analysis = LinearizationAnalysis().check().prepare(System(elements=[CartElement('cart')])).run()
        >>> analysis.B_.tolist(), analysis.controllability_
        ([[0.0], [1.0]], 2)
    """

    def __init__(
        self,
        controls: Sequence[str] | None = None,
        outputs: Sequence[str] | None = None,
        at: Mapping[str, float] | None = None,
        *,
        state_weights: npt.ArrayLike | None = None,
        control_weights: npt.ArrayLike | None = None,
    ) -> None:
        self.controls = controls
        self.outputs = outputs
        self.at = at
        self.state_weights = state_weights
        self.control_weights = control_weights

    def _check(self) -> Self:
        """Check that the weights of the regulator are given together."""
        if (self.state_weights is None) != (self.control_weights is None):
            error_msg = (
                f'{type(self).__name__} computes the regulator from both state_weights and control_weights. Give both, '
                'or neither.'
            )
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the system kind."""
        return DataKind.SYSTEM

    def _prepare(self, source: Source) -> Self:
        """Compile the system of the source, checking that it runs in continuous time."""
        problem = self._compiled_problem(self._read_system(source), 'Linearization')
        if problem.form != Form.ODE:
            error_msg = (
                f'{type(self).__name__} takes systems under continuous time, but the system is a {problem.form.value}.'
            )
            raise ValidationError(error_msg)
        self.problem_ = problem
        return self

    def _run(self, inputs: object = None) -> Self:
        """Derive the state-space matrices, their ranks, and the gain of the regulator."""
        problem = self.problem_
        adaptive = problem.adaptive_parameters
        chosen = (
            [name for name, item in adaptive.items() if item.bounds is not None]
            if self.controls is None
            else self.controls
        )
        inputs = self._checked_names(chosen, list(adaptive), 'adaptive parameters')
        dynamics = StabilityAnalysis._dynamics_of(problem)
        states = [name for name in dynamics.names if name not in inputs]
        observed = self._checked_names(
            states if self.outputs is None else self.outputs,
            [*states, *inputs],
            'states or controls',
        )
        values = self._initial_values(problem)
        values.update(self.at or {})
        by_name = dict(zip(dynamics.names, dynamics.symbols, strict=True))
        rates = dict(zip(dynamics.names, dynamics.rates, strict=True))
        point = {by_name[name]: float(value) for name, value in values.items() if name in by_name}
        point.update(dynamics.parameters)
        point[dynamics.time] = 0.0
        self.A_ = self._evaluated([[rates[row].diff(by_name[column]) for column in states] for row in states], point)
        self.B_ = self._evaluated([[rates[row].diff(by_name[column]) for column in inputs] for row in states], point)
        self.C_ = np.array([[float(row == column) for column in states] for row in observed]).reshape(
            len(observed),
            len(states),
        )
        self.D_ = np.array([[float(row == column) for column in inputs] for row in observed]).reshape(
            len(observed),
            len(inputs),
        )
        self.states_ = tuple(states)
        self.controls_ = tuple(inputs)
        self.outputs_ = tuple(observed)
        self.controllability_ = self._controllability(self.A_, self.B_)
        self.observability_ = self._controllability(self.A_.T, self.C_.T)
        self.gain_ = (
            None
            if self.state_weights is None
            else self._lqr(self.A_, self.B_, self.state_weights, self.control_weights)  # type: ignore[arg-type]
        )
        return self

    @staticmethod
    def _initial_values(problem: Problem) -> dict[str, float]:
        """Return the initial value of every state component by name."""
        return {
            name: float(problem.initial_state[entry.start + column])
            for entry in problem.states.entries
            for column, name in enumerate(entry.component_names)
        }

    @staticmethod
    def _checked_names(names: Sequence[str], known: Sequence[str], description: str) -> list[str]:
        """Return names after checking that each is one of the known names."""
        unknown = [name for name in names if name not in known]
        if unknown:
            error_msg = f'{", ".join(unknown)} are not {description}. Available names are: {", ".join(known)}.'
            raise AnalysisError(error_msg)
        return list(names)

    @staticmethod
    def _evaluated(rows: list[list[Expression]], values: Mapping[Expression, float]) -> Matrix:
        """Return a matrix of expressions evaluated at values of their symbols."""
        columns = len(rows[0]) if rows else 0
        return np.array([[float(SR(item).subs(values)) for item in row] for row in rows]).reshape(len(rows), columns)

    @staticmethod
    def _controllability(state: Matrix, inputs: Matrix) -> int:
        """Return the rank of the controllability matrix `[B, AB, ..., A^(n-1) B]`."""
        blocks = [inputs]
        for _ in range(state.shape[0] - 1):
            blocks.append(state @ blocks[-1])
        return int(np.linalg.matrix_rank(np.hstack(blocks)))

    @staticmethod
    def _lqr(state: Matrix, inputs: Matrix, state_weights: npt.ArrayLike, control_weights: npt.ArrayLike) -> Matrix:
        """Return the gain `K = R^-1 B' P` of the linear-quadratic regulator, from the Riccati solution `P` of SciPy."""
        weights = np.atleast_2d(np.asarray(control_weights, dtype=float))
        riccati = solve_continuous_are(state, inputs, np.atleast_2d(np.asarray(state_weights, dtype=float)), weights)
        gain: Matrix = np.linalg.solve(weights, inputs.T @ riccati)
        return gain
