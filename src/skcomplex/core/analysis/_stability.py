"""Derive fixed points, Jacobians, stability, nullclines, and divergence of systems symbolically."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, matrix, solve
from scipy.optimize import root

from ..base import AnalysisError, Expression, Form, components_of, declare_domains
from ..compile import Problem, compile_system
from ..model import System
from ._groups import expand_groups

ZERO = 1e-9


@dataclass(frozen=True)
class Dynamics:
    """First-order equations of a system: its state symbols, their right sides, and its parameter values.

    Args:
        problem:
            The compiled problem.

        names:
            The state component names, in the order of the symbols.

        symbols:
            The state symbols.

        rates:
            The right side of every state's equation: the time derivative for flows and the next value for maps.

        parameters:
            The current value of every parameter symbol.

        discrete:
            Whether the system is a map.
    """

    problem: Problem
    names: list[str]
    symbols: list[Expression]
    rates: list[Expression]
    parameters: dict[Expression, float]
    discrete: bool

    @property
    def time(self) -> Expression:
        """Time symbol of the equations."""
        return self.problem.time


def dynamics_of(system: System | Problem, parameters: Mapping[str, float] | None = None) -> Dynamics:
    """Return the first-order equations of a system with the current values of its parameters.

    Args:
        system:
            The system, or a problem compiled from one.

        parameters:
            Values of parameters by dotted path that replace the system's values.

    Returns:
        dynamics:
            The equations.

    Raises:
        AnalysisError:
            The system is not an ODE or a discrete map, has callable laws or sums over members into single elements,
            or has a group of more than 64 members.
    """
    problem = system if isinstance(system, Problem) else compile_system(system)
    if problem.form not in {Form.ODE, Form.DISCRETE}:
        error_msg = (
            f'Only ODE systems and maps have a vector field to analyse, but the system is a {problem.form.value}.'
        )
        raise AnalysisError(error_msg)
    if problem.has_callables or problem.has_group_sums:
        error_msg = (
            'The system has callable laws or sums over group members into single elements, which have no symbolic '
            'vector field. Write the sums as reductions, or analyse one member.'
        )
        raise AnalysisError(error_msg)
    if problem.delays:
        error_msg = (
            'The system reads delayed states, so its state space is infinite-dimensional and it has no vector field '
            'of finitely many states. Analyse its trajectories instead.'
        )
        raise AnalysisError(error_msg)
    values = problem.initial_parameters.copy()
    for name, value in (parameters or {}).items():
        if name not in problem.parameters:
            error_msg = f'{name} is not a parameter. Available parameters are: {", ".join(problem.parameters.names)}.'
            raise AnalysisError(error_msg)
        values[problem.parameters.entry(name).indices] = value
    current = {}
    for entry in problem.parameters:
        for column, item in enumerate(components_of(problem.symbols[entry.name])):
            current[item] = float(values[entry.start + column])
    names, symbols, rates, members = expand_groups(problem, values)
    return Dynamics(
        problem=problem,
        names=names,
        symbols=symbols,
        rates=rates,
        parameters={**current, **members},
        discrete=problem.form is Form.DISCRETE,
    )


def jacobian(system: System | Problem) -> Expression:
    """Return the Jacobian matrix of a system's vector field, or of its map, in the order of its state components.

    Args:
        system:
            The system, or a problem compiled from one.

    Returns:
        jacobian:
            The Sage matrix of partial derivatives of the right sides by the states.

    Raises:
        AnalysisError:
            The system is not an ODE or a discrete map, has callable laws, or has a group above the member limit.

    Examples:
        >>> from skcomplex.core.analysis import jacobian
        >>> from skcomplex.models.biology.ecology import logistic_growth
        >>> jacobian(logistic_growth()).dimensions()
        (1, 1)
    """
    dynamics = dynamics_of(system)
    return matrix(SR, [[rate.diff(item) for item in dynamics.symbols] for rate in dynamics.rates])


def fixed_points(system: System | Problem, parameters: Mapping[str, float] | None = None) -> list[dict[str, Any]]:
    """Return the fixed points of a flow, where every rate is 0, or of a map, where every state equals its next value.

    Sage solves the equations symbolically. When it finds no closed form, roots are searched numerically from a grid
    of starting points with the parameters at their values.

    Args:
        system:
            The system, or a problem compiled from one.

        parameters:
            Values of parameters by dotted path, substituted before solving. Without them, symbolic solutions keep the
            parameters as symbols.

    Returns:
        points:
            One mapping from state component name to its value per fixed point.

    Raises:
        AnalysisError:
            The system is not an ODE or a discrete map, has callable laws, or has a group above the member limit.

    Examples:
        >>> from skcomplex.core.analysis import fixed_points
        >>> from skcomplex.models.biology.ecology import logistic_growth
        >>> sorted(str(point['population.n']) for point in fixed_points(logistic_growth()))
        ['0', 'K_growth']
    """
    dynamics = dynamics_of(system, parameters)
    substitution = {item: value for item, value in dynamics.parameters.items() if parameters}
    declare_domains([*dynamics.symbols, *dynamics.parameters], dynamics.parameters)
    equations = [
        (rate - item if dynamics.discrete else rate).subs(substitution) == 0
        for item, rate in zip(dynamics.symbols, dynamics.rates, strict=True)
    ]
    try:
        solutions = solve(equations, dynamics.symbols, solution_dict=True)
    except (TypeError, ValueError, NotImplementedError):
        solutions = []
    points = [
        {name: solution[item] for name, item in zip(dynamics.names, dynamics.symbols, strict=True)}
        for solution in solutions
        if _is_point(solution, dynamics.symbols)
    ]
    return points or _numeric_points(dynamics)


def _is_point(solution: Mapping[Expression, Expression], symbols: list[Expression]) -> bool:
    """Check that a solution fixes every state, without the free parameters `r1`, `r2`, ...

    of solution families.
    """
    free = {str(item) for value in solution.values() for item in SR(value).variables()}
    return all(item in solution for item in symbols) and not any(re.fullmatch(r'r\d+', name) for name in free)


def _numeric_points(dynamics: Dynamics) -> list[dict[str, Any]]:
    """Return fixed points found numerically from a grid of starting points."""
    values = [
        rate.subs(dynamics.parameters) - (item if dynamics.discrete else 0)
        for item, rate in zip(dynamics.symbols, dynamics.rates, strict=True)
    ]
    functions = [value.function(*dynamics.symbols) for value in values]
    found: list[npt.NDArray[np.float64]] = []
    rng = np.random.default_rng(0)
    for start in rng.uniform(-10, 10, (50, len(dynamics.symbols))):
        result = root(lambda point: [float(function(*point)) for function in functions], start)
        if result.success and not any(np.allclose(result.x, other, atol=1e-6) for other in found):
            found.append(result.x)
    return [dict(zip(dynamics.names, map(float, point), strict=True)) for point in found]


@dataclass(frozen=True)
class Stability:
    """Linear stability of a fixed point.

    Args:
        eigenvalues:
            The eigenvalues of the Jacobian at the point, evaluated at the parameter values.

        classification:
            `stable node`, `stable focus`, `unstable node`, `unstable focus`, `saddle`, `centre`, or `marginal` for
            flows, and `stable`, `unstable`, `saddle`, or `marginal` for maps.
    """

    eigenvalues: npt.NDArray[np.complex128]
    classification: str


def stability(
    system: System | Problem,
    point: Mapping[str, Any],
    parameters: Mapping[str, float] | None = None,
) -> Stability:
    """Return the eigenvalues of the Jacobian at a fixed point and the point's stability.

    Args:
        system:
            The system, or a problem compiled from one.

        point:
            A mapping from state component name to its value, symbolic or numeric.

        parameters:
            Values of parameters by dotted path that replace the system's values.

    Returns:
        stability:
            The eigenvalues and the classification.

    Raises:
        AnalysisError:
            The system is not an ODE or a discrete map, has callable laws, or has a group above the member limit.

    Examples:
        >>> from skcomplex.core.analysis import stability
        >>> from skcomplex.models.biology.ecology import logistic_growth
        >>> stability(logistic_growth(), {'population.n': 100.0}).classification
        'stable node'
    """
    dynamics = dynamics_of(system, parameters)
    location = {item: SR(point[name]) for name, item in zip(dynamics.names, dynamics.symbols, strict=True)}
    numbers = {**location, **dynamics.parameters}
    values = matrix(SR, [[rate.diff(item) for item in dynamics.symbols] for rate in dynamics.rates]).subs(location)
    numeric = np.array([[complex(SR(entry).subs(numbers)) for entry in row] for row in values.rows()], dtype=complex)
    eigenvalues = np.linalg.eigvals(numeric)
    return Stability(eigenvalues, _classify(eigenvalues, discrete=dynamics.discrete))


def _classify_map(eigenvalues: npt.NDArray[np.complex128]) -> str:
    """Return the stability class of a fixed point of a map from its eigenvalues."""
    moduli = np.abs(eigenvalues)
    if np.all(moduli < 1 - ZERO):
        return 'stable'
    if np.all(moduli > 1 + ZERO):
        return 'unstable'
    return 'saddle' if np.any(moduli < 1 - ZERO) and np.any(moduli > 1 + ZERO) else 'marginal'


def _classify(eigenvalues: npt.NDArray[np.complex128], *, discrete: bool) -> str:
    """Return the stability class of a fixed point from its eigenvalues."""
    if discrete:
        return _classify_map(eigenvalues)
    scale = max(1.0, float(np.abs(eigenvalues).max()))
    real = eigenvalues.real / scale
    oscillating = bool(np.any(np.abs(eigenvalues.imag) > ZERO * scale))
    if np.all(real < -ZERO):
        return 'stable focus' if oscillating else 'stable node'
    if np.all(real > ZERO):
        return 'unstable focus' if oscillating else 'unstable node'
    if np.any(real > ZERO) and np.any(real < -ZERO):
        return 'saddle'
    return 'centre' if oscillating and np.all(np.abs(real) <= ZERO) else 'marginal'


def nullclines(system: System | Problem) -> dict[str, Expression]:
    """Return the nullcline of every state component, the set where its rate is 0 or where it is its own next value.

    Args:
        system:
            The system, or a problem compiled from one.

    Returns:
        nullclines:
            A mapping from state component name to its nullcline as a Sage relation.

    Raises:
        AnalysisError:
            The system is not an ODE or a discrete map, has callable laws, or has a group above the member limit.

    Examples:
        >>> from skcomplex.core.analysis import nullclines
        >>> from skcomplex.models.biology.ecology import exponential_growth
        >>> nullclines(exponential_growth())
        {'population.n': n_population*r_growth == 0}
    """
    dynamics = dynamics_of(system)
    return {
        name: (rate - item if dynamics.discrete else rate) == 0
        for name, item, rate in zip(dynamics.names, dynamics.symbols, dynamics.rates, strict=True)
    }


def divergence(system: System | Problem) -> Expression:
    """Return the divergence of a flow's vector field, the trace of its Jacobian, 0 for volume-preserving flows.

    Args:
        system:
            The system, or a problem compiled from one.

    Returns:
        divergence:
            The divergence as a Sage expression.

    Raises:
        AnalysisError:
            The system is not an ODE or a discrete map, has callable laws, or has a group above the member limit.

    Examples:
        >>> from skcomplex.core.analysis import divergence
        >>> from skcomplex.models.physics.mechanics import harmonic_oscillator
        >>> divergence(harmonic_oscillator())
        0
    """
    return jacobian(system).trace().simplify_full()
