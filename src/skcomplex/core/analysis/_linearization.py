"""Linearise a system into state-space matrices.

The module also gives its controllability, observability, and optimal gain.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression
from scipy.linalg import solve_continuous_are

from ..engine.problems import ProblemSpec
from ..errors import AnalysisError, ValidationError
from ..specs import AnalysisTags
from ..types import Array, DataKind, FormKind
from ._base import _composite, _read_system
from ._stability import StabilityResult
from ._utils import compile_problem
from .types import Source

TAGS = AnalysisTags(DataKind.SYSTEM)


@dataclass(frozen=True)
class LinearizationResult:
    """Linear state-space system of a system near an operating point.

    It comes with its ranks and optimal gain.

    Args:
        problem:
            The compiled problem.

        A:
            The state matrix, the Jacobian of the rates by the states.

        B:
            The input matrix, the Jacobian of the rates by the controls.

        C:
            The output matrix, the Jacobian of the outputs by the states.

        D:
            The feedthrough matrix, the Jacobian of the outputs by the controls.

        states:
            The names of the state components, in the order of the rows of `A`.

        controls:
            The names of the controls, in the order of the columns of `B`.

        outputs:
            The names of the outputs, in the order of the rows of `C`.

        controllability:
            The rank of the controllability matrix.

        observability:
            The rank of the observability matrix.

        gain:
            The gain of the regulator, one row per control and one column per state, or `None` without weights.
    """

    problem: ProblemSpec
    A: Array
    B: Array
    C: Array
    D: Array
    states: tuple[str, ...]
    controls: tuple[str, ...]
    outputs: tuple[str, ...]
    controllability: int
    observability: int
    gain: Array | None


def linearization(
    source: Source,
    *,
    controls: Sequence[str] | None = None,
    outputs: Sequence[str] | None = None,
    at: Mapping[str, float] | None = None,
    state_weights: npt.ArrayLike | None = None,
    control_weights: npt.ArrayLike | None = None,
) -> LinearizationResult:
    """Return the linear state-space system of a system near an operating point.

    The system is `x' = A x + B u`, `y = C x + D u`. The matrices are the Jacobians of the rates and of the outputs,
    derived symbolically and evaluated at the operating point. The controls are adaptive parameters, named by their
    dotted names, and the outputs are states or controls. The system is controllable when the rank of `[B, AB, ...,
    A^(n-1) B]` equals the number of states, and observable when the rank of `[C; CA; ...; C A^(n-1)]` does. With
    weights of the states and the controls, the gain `K` of the linear-quadratic regulator `u = -K x` minimises `∫ (x' Q
    x + u' R u) dt` over an infinite horizon, from the continuous algebraic Riccati equation that SciPy solves.

    Args:
        source:
            The system, its compiled problem, or a trajectory of it.

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

    Returns:
        result:
            The state-space matrices, their ranks, and the gain of the regulator.

    Raises:
        AnalysisError:
            Only one of the weights is given, or a control or an output is unknown.

        ValidationError:
            The system does not run in continuous time.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.analysis import linearization
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.laws import RateLaw
        >>> from skcomplex.core.specs import ParameterSpec, StateSpec
        >>> from skcomplex.core.components.systems import CompositeSystem
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
        >>> result = linearization(CompositeSystem(elements=[CartElement('cart')]))
        >>> result.B.tolist(), result.controllability
        ([[0.0], [1.0]], 2)
    """
    source = _composite(source)
    if (state_weights is None) != (control_weights is None):
        error_msg = (
            'linearization computes the regulator from both state_weights and control_weights. Give both, or neither.'
        )
        raise AnalysisError(error_msg)
    problem = compile_problem(_read_system(source), 'Linearization')
    if problem.form != FormKind.ODE:
        error_msg = f'linearization takes systems under continuous time, but the system is a {problem.form}.'
        raise ValidationError(error_msg)
    adaptive = problem.adaptive_parameters
    chosen = [name for name, item in adaptive.items() if item.bounds is not None] if controls is None else controls
    inputs = _checked_names(chosen, list(adaptive), 'adaptive parameters')
    dynamics = StabilityResult._get_dynamics_of(problem)
    states = [name for name in dynamics.names if name not in inputs]
    observed = _checked_names(states if outputs is None else outputs, [*states, *inputs], 'states or controls')
    values = _initial_values(problem)
    values.update(at or {})
    by_name = dict(zip(dynamics.names, dynamics.symbols, strict=True))
    rates = dict(zip(dynamics.names, dynamics.rates, strict=True))
    point = {by_name[name]: float(value) for name, value in values.items() if name in by_name}
    point.update(dynamics.parameters)
    point[dynamics.time] = 0.0
    state_matrix = _evaluated([[rates[row].diff(by_name[column]) for column in states] for row in states], point)
    input_matrix = _evaluated([[rates[row].diff(by_name[column]) for column in inputs] for row in states], point)
    output_matrix = np.array([[float(row == column) for column in states] for row in observed]).reshape(
        len(observed),
        len(states),
    )
    feedthrough = np.array([[float(row == column) for column in inputs] for row in observed]).reshape(
        len(observed),
        len(inputs),
    )
    gain = (
        None
        if state_weights is None or control_weights is None
        else _lqr(state_matrix, input_matrix, state_weights, control_weights)
    )
    return LinearizationResult(
        problem=problem,
        A=state_matrix,
        B=input_matrix,
        C=output_matrix,
        D=feedthrough,
        states=tuple(states),
        controls=tuple(inputs),
        outputs=tuple(observed),
        controllability=_controllability(state_matrix, input_matrix),
        observability=_controllability(state_matrix.T, output_matrix.T),
        gain=gain,
    )


def _initial_values(problem: ProblemSpec) -> dict[str, float]:
    """Return the initial value of every state component by name."""
    return {
        name: float(problem.initial_state[entry.start + column])
        for entry in problem.states.entries
        for column, name in enumerate(entry.component_names)
    }


def _checked_names(names: Sequence[str], known: Sequence[str], description: str) -> list[str]:
    """Return names after checking that each is one of the known names."""
    unknown = [name for name in names if name not in known]
    if unknown:
        error_msg = f'{", ".join(unknown)} are not {description}. Available names are: {", ".join(known)}.'
        raise AnalysisError(error_msg)
    return list(names)


def _evaluated(rows: list[list[Expression]], values: Mapping[Expression, float]) -> Array:
    """Return a matrix of expressions evaluated at values of their symbols."""
    columns = len(rows[0]) if rows else 0
    return np.array([[float(SR(item).subs(values)) for item in row] for row in rows]).reshape(len(rows), columns)


def _controllability(state: Array, inputs: Array) -> int:
    """Return the rank of the controllability matrix `[B, AB, ..., A^(n-1) B]`."""
    blocks = [inputs]
    for _ in range(state.shape[0] - 1):
        blocks.append(state @ blocks[-1])
    return int(np.linalg.matrix_rank(np.hstack(blocks)))


def _lqr(state: Array, inputs: Array, state_weights: npt.ArrayLike, control_weights: npt.ArrayLike) -> Array:
    """Return the gain `K = R^-1 B' P` of the linear-quadratic regulator.

    `P` is the Riccati solution of SciPy.
    """
    weights = np.atleast_2d(np.asarray(control_weights, dtype=float))
    riccati = solve_continuous_are(state, inputs, np.atleast_2d(np.asarray(state_weights, dtype=float)), weights)
    gain: Array = np.linalg.solve(weights, inputs.T @ riccati)
    return gain
