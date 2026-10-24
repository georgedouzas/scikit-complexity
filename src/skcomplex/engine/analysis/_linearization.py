"""Linearise systems into state-space matrices and analyse their controllability, observability, and optimal gains."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR
from scipy.linalg import solve_continuous_are

from ..base import AnalysisError, Expression, Form, ValidationError
from ..compile import Problem, compile_system
from ..system import System
from ._stability import dynamics_of

type Matrix = npt.NDArray[np.float64]


def _initial_values(problem: Problem) -> dict[str, float]:
    """Return the initial value of every state component by name."""
    return {
        name: float(problem.initial_state[entry.start + column])
        for entry in problem.states
        for column, name in enumerate(entry.component_names)
    }


def _checked_names(names: Sequence[str], known: Sequence[str], description: str) -> list[str]:
    """Return names after checking that each is one of the known names."""
    unknown = [name for name in names if name not in known]
    if unknown:
        error_msg = f'{", ".join(unknown)} are not {description}. Available names are: {", ".join(known)}.'
        raise AnalysisError(error_msg)
    return list(names)


def _evaluated(rows: list[list[Expression]], values: Mapping[Expression, float]) -> Matrix:
    """Return a matrix of expressions evaluated at values of their symbols."""
    columns = len(rows[0]) if rows else 0
    return np.array([[float(SR(item).subs(values)) for item in row] for row in rows]).reshape(len(rows), columns)


@dataclass(frozen=True)
class StateSpace:
    """Linear state-space model `x' = A x + B u`, `y = C x + D u` of a system near an operating point.

    Args:
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
            The names of the controls, the adaptive parameters, in the order of the columns of `B`.

        outputs:
            The names of the outputs, in the order of the rows of `C`.
    """

    A: Matrix
    B: Matrix
    C: Matrix
    D: Matrix
    states: tuple[str, ...]
    controls: tuple[str, ...]
    outputs: tuple[str, ...]


def controllability(A: npt.ArrayLike, B: npt.ArrayLike) -> int:
    """Return the rank of the controllability matrix `[B, AB, ..., A^(n-1) B]` of a linear system.

    The system is controllable when the rank equals the number of states.

    Args:
        A:
            The state matrix.

        B:
            The input matrix.

    Returns:
        rank:
            The rank of the controllability matrix.

    Examples:
        >>> from skcomplex.core.analysis import controllability
        >>> controllability([[0, 1], [-1, 0]], [[0], [1]])
        2
    """
    state = np.atleast_2d(np.asarray(A, dtype=float))
    inputs = np.asarray(B, dtype=float).reshape(state.shape[0], -1)
    blocks = [inputs]
    for _ in range(state.shape[0] - 1):
        blocks.append(state @ blocks[-1])
    return int(np.linalg.matrix_rank(np.hstack(blocks)))


def observability(A: npt.ArrayLike, C: npt.ArrayLike) -> int:
    """Return the rank of the observability matrix `[C; CA; ...; C A^(n-1)]` of a linear system.

    The system is observable when the rank equals the number of states.

    Args:
        A:
            The state matrix.

        C:
            The output matrix.

    Returns:
        rank:
            The rank of the observability matrix.

    Examples:
        >>> from skcomplex.core.analysis import observability
        >>> observability([[0, 1], [-1, 0]], [[0, 1]])
        2
    """
    state = np.atleast_2d(np.asarray(A, dtype=float))
    return controllability(state.T, np.asarray(C, dtype=float).reshape(-1, state.shape[0]).T)


def lqr(A: npt.ArrayLike, B: npt.ArrayLike, Q: npt.ArrayLike, R: npt.ArrayLike) -> Matrix:
    """Return the gain `K` of the linear-quadratic regulator `u = -K x` of a linear system.

    The gain minimises `∫ (x' Q x + u' R u) dt` over an infinite horizon, with `K = R^-1 B' P` from the solution `P`
    of the continuous algebraic Riccati equation, which SciPy solves.

    Args:
        A:
            The state matrix.

        B:
            The input matrix.

        Q:
            The weight of the states, a symmetric positive semidefinite matrix.

        R:
            The weight of the controls, a symmetric positive definite matrix.

    Returns:
        gain:
            The gain matrix, one row per control and one column per state.

    Examples:
        >>> from skcomplex.core.analysis import lqr
        >>> lqr([[0.0]], [[1.0]], [[1.0]], [[1.0]]).round(6).tolist()
        [[1.0]]
    """
    state = np.atleast_2d(np.asarray(A, dtype=float))
    inputs = np.asarray(B, dtype=float).reshape(state.shape[0], -1)
    weights = np.atleast_2d(np.asarray(R, dtype=float))
    riccati = solve_continuous_are(state, inputs, np.atleast_2d(np.asarray(Q, dtype=float)), weights)
    gain: Matrix = np.linalg.solve(weights, inputs.T @ riccati)
    return gain


def linearize(
    system: System,
    controls: Sequence[str] | None = None,
    outputs: Sequence[str] | None = None,
    at: Mapping[str, float] | None = None,
) -> StateSpace:
    """Return the linearisation of a system near an operating point as state-space matrices.

    The matrices are the Jacobians of the rates and of the outputs, derived symbolically and evaluated at the
    operating point. The controls are adaptive parameters, named by their dotted names, and the outputs are states or
    controls.

    Args:
        system:
            The system under continuous time.

        controls:
            The dotted names of the adaptive parameters that are inputs, or `None` for the bounded ones.

        outputs:
            The dotted names of the states and controls that are outputs, or `None` for every state.

        at:
            The values of states and controls at the operating point, by dotted name, or `None` for the initial values.

    Returns:
        state_space:
            The matrices with the names of their states, controls, and outputs.

    Raises:
        ValidationError:
            The system is not valid, or not under continuous time.

        AnalysisError:
            A name is not an adaptive parameter, a state, or a control, or the system has no symbolic vector field.

    Examples:
        >>> from skcomplex.core.analysis import controllability, linearize
        >>> from skcomplex.core.base import parameter
        >>> from skcomplex.core.components import Element
        >>> from skcomplex.core.laws import Rate
        >>> from skcomplex.core.system import System
        >>> Cart = Element.define(
        ...     'Cart',
        ...     states={'x': 0.0, 'v': 0.0},
        ...     parameters={'force': parameter(0.0, adaptive=True, bounds=(-1, 1))},
        ...     laws=lambda s: [Rate(s.x, s.v), Rate(s.v, s.force)],
        ... )
        >>> model = linearize(System(components=[Cart('cart')]))
        >>> model.B.tolist(), controllability(model.A, model.B)
        ([[0.0], [1.0]], 2)
    """
    problem = compile_system(system)
    if problem.form is not Form.ODE:
        error_msg = f'linearize takes systems under continuous time, but the system is a {problem.form.value}.'
        raise ValidationError(error_msg)
    adaptive = problem.adaptive_parameters
    chosen = [name for name, item in adaptive.items() if item.bounds is not None] if controls is None else controls
    inputs = _checked_names(chosen, list(adaptive), 'adaptive parameters')
    dynamics = dynamics_of(problem)
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
    output_matrix = np.array([[float(row == column) for column in states] for row in observed]).reshape(-1, len(states))
    feedthrough = np.array([[float(row == column) for column in inputs] for row in observed]).reshape(-1, len(inputs))
    return StateSpace(
        state_matrix,
        input_matrix,
        output_matrix,
        feedthrough,
        tuple(states),
        tuple(inputs),
        tuple(observed),
    )
