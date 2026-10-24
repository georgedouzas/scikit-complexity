"""Measure chaos with the Lyapunov spectrum of flows and maps from their tangent dynamics."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Mapping
from typing import Any

import numba
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, dirac_delta, sgn

from ..compile import SCALAR_DIALECT, Problem, print_expressions
from ..system import System
from ._stability import Dynamics, dynamics_of

FUNCTIONS: dict[str, Callable[..., Any]] = {}


def _without_kinks(expression: Any) -> Any:
    """Return an expression with every `abs(u)` written as `u sgn(u)`, whose derivative is `sgn(u)`, not `u / |u|`."""
    wild = SR.wild(0)
    return SR(expression).subs({abs(wild): wild * sgn(wild)})


def _jacobian(dynamics: Dynamics) -> list[Any]:
    """Return the entries of the Jacobian of the rates row by row, with the kinks of absolute values as signs."""
    wild = SR.wild(0)
    rates = [_without_kinks(rate) for rate in dynamics.rates]
    return [rate.diff(item).subs({dirac_delta(wild): 0}) for rate in rates for item in dynamics.symbols]


def _function(
    name: str,
    dynamics: Dynamics,
    expressions: list[Any],
    shape: tuple[int, ...],
) -> Callable[..., Any]:
    """Return a Numba function `f(t, x, p)` that fills an array of a shape, compiled once per printed source.

    The parameters are read from `p` in the order of `dynamics.parameters`, so models of one structure share the
    compiled function whatever their parameter values.
    """
    names = (
        {item: f'x[{index}]' for index, item in enumerate(dynamics.symbols)}
        | {item: f'p[{index}]' for index, item in enumerate(dynamics.parameters)}
        | {dynamics.time: 't'}
    )
    assignments, codes = print_expressions([SR(item) for item in expressions], SCALAR_DIALECT, lambda item: names[item])
    lines = [f'def {name}(t, x, p):', f'    out = np.zeros({shape})']
    lines += [f'    {bound} = {code}' for bound, code in assignments]
    lines += [
        f'    out[{np.unravel_index(index, shape) if len(shape) > 1 else index}] = {code}'
        for index, code in enumerate(codes)
    ]
    lines.append('    return out')
    source = '\n'.join(lines)
    if source not in FUNCTIONS:
        namespace: dict[str, Any] = {'np': np, 'math': math}
        exec(source, namespace)  # noqa: S102  # nosec B102: runs code printed from the model's own expressions
        FUNCTIONS[source] = numba.njit(namespace[name])
    return FUNCTIONS[source]


@numba.njit
def _flow_spectrum(
    f: Any,
    jac: Any,
    x: Any,
    p: Any,
    n: int,
    dt: float,
    steps: int,
    skip: int,
) -> Any:
    """Integrate a flow and its tangent vectors with RK4, re-orthonormalising every step, and average the growth."""
    q = np.eye(x.size)[:, :n].copy()
    sums = np.zeros(n)
    t = 0.0
    for step in range(skip + steps):
        k1 = f(t, x, p)
        l1 = jac(t, x, p) @ q
        k2 = f(t + dt / 2, x + dt / 2 * k1, p)
        l2 = jac(t + dt / 2, x + dt / 2 * k1, p) @ (q + dt / 2 * l1)
        k3 = f(t + dt / 2, x + dt / 2 * k2, p)
        l3 = jac(t + dt / 2, x + dt / 2 * k2, p) @ (q + dt / 2 * l2)
        k4 = f(t + dt, x + dt * k3, p)
        l4 = jac(t + dt, x + dt * k3, p) @ (q + dt * l3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        q = q + dt / 6 * (l1 + 2 * l2 + 2 * l3 + l4)
        q, r = np.linalg.qr(q)
        q = np.ascontiguousarray(q)
        if step >= skip:
            sums += np.log(np.abs(np.diag(r)))
        t += dt
    return sums / (steps * dt)


@numba.njit
def _map_spectrum(f: Any, jac: Any, x: Any, p: Any, n: int, steps: int, skip: int) -> Any:
    """Iterate a map and its tangent vectors, re-orthonormalising every step, and average the growth."""
    q = np.eye(x.size)[:, :n].copy()
    sums = np.zeros(n)
    for step in range(skip + steps):
        q = jac(float(step), x, p) @ q
        x = f(float(step), x, p)
        q, r = np.linalg.qr(q)
        q = np.ascontiguousarray(q)
        if step >= skip:
            sums += np.log(np.abs(np.diag(r)))
    return sums / steps


def lyapunov_spectrum(
    system: System | Problem,
    n: int | None = None,
    *,
    until: float | None = None,
    dt: float = 0.01,
    transient: float | None = None,
    parameters: Mapping[str, float] | None = None,
) -> npt.NDArray[np.float64]:
    """Return the largest `n` Lyapunov exponents of a flow or a map, from the system's initial state.

    The tangent dynamics use the symbolic Jacobian, in which `abs(u)` has the derivative `sign(u)`. The compiled
    functions take the parameters as an argument, so a sweep over parameter values compiles them once. Tangent vectors
    are re-orthonormalised by QR after every step, as in the Benettin algorithm, and the exponents are the average
    logarithmic growth of the diagonal of `R`. A positive largest exponent indicates chaos.

    Args:
        system:
            The system, or a problem compiled from one.

        n:
            The number of exponents, or `None` for all of them.

        until:
            The time averaged over, or the number of iterations of a map. It defaults to 1000 for flows and 10,000 for
            maps.

        dt:
            The fixed step of the RK4 integration of flows.

        transient:
            The time, or the number of iterations, run before averaging. It defaults to a tenth of `until`.

        parameters:
            Values of parameters by dotted path that replace the system's values.

    Returns:
        exponents:
            The exponents in decreasing order, per unit of time for flows and per iteration for maps.

    Raises:
        AnalysisError:
            The system is not an ODE or a discrete map, has callable laws, or has a group above the member limit.

    Examples:
        >>> from skcomplex.core.analysis import lyapunov_spectrum
        >>> from skcomplex.models.mathematics.dynamical_systems import logistic_map
        >>> float(lyapunov_spectrum(logistic_map(r=4.0), until=20000)[0].round(2))
        0.69
    """
    dynamics = dynamics_of(system, parameters)
    size = len(dynamics.symbols)
    count = size if n is None else n
    problem = dynamics.problem
    indices = {
        name: entry.start + column for entry in problem.states for column, name in enumerate(entry.generic_names)
    }
    start = np.asarray([problem.initial_state[indices[name]] for name in dynamics.names], dtype=float)
    f = _function('field', dynamics, dynamics.rates, (size,))
    jac = _function('jacobian', dynamics, _jacobian(dynamics), (size, size))
    values = np.asarray([float(value) for value in dynamics.parameters.values()], dtype=float)
    if dynamics.discrete:
        steps = int(until if until is not None else 10_000)
        skip = int(transient if transient is not None else steps // 10)
        exponents = _map_spectrum(f, jac, start, values, count, steps, skip)
    else:
        total = until if until is not None else 1000.0
        steps = round(total / dt)
        skip = round((transient if transient is not None else total / 10) / dt)
        exponents = _flow_spectrum(f, jac, start, values, count, dt, steps, skip)
    return np.sort(np.asarray(exponents))[::-1]
