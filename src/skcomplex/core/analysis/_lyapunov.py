"""Measure chaos with Lyapunov exponents: the spectrum of a system from its tangent dynamics, or the largest of data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import nolds
import numba
import numpy as np
from sage.all__sagemath_symbolics import SR, Expression, dirac_delta, sgn

from ..components.system import System
from ..engine.problems import BaseProblemSpec, ExpressionPrinter, ProblemSpec, SourceLoader
from ..engine.problems._dynamics_spec import _DynamicsSpec
from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import Array, VectorFunction
from ._base import _check_choice, _check_settings, _read_series, _read_system
from ._stability import StabilityAnalysis
from .types import Source

KINDS = ('spectrum', 'largest')
SETTINGS = {
    'spectrum': {'n_exponents', 'until', 'dt', 'transient', 'parameters'},
    'largest': {'variables', 'dt', 'dimension', 'delay', 'min_separation', 'trajectory_length', 'skip'},
}
"""Settings every kind reads, besides `kind`."""

FUNCTIONS: dict[str, VectorFunction] = {}


@numba.njit
def _flow_spectrum(
    f: VectorFunction,
    jac: VectorFunction,
    x: Array,
    p: Array,
    n: int,
    dt: float,
    steps: int,
    skip: int,
) -> Array:
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
def _map_spectrum(f: VectorFunction, jac: VectorFunction, x: Array, p: Array, n: int, steps: int, skip: int) -> Array:
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


@dataclass(frozen=True)
class LyapunovResult:
    """Lyapunov exponents of a system, or the largest exponent of every recorded series.

    Args:
        problem:
            The compiled problem of the spectrum of a system, or `None` for an array of exponents or for `'largest'`.

        exponents:
            The exponents of the spectrum in decreasing order, per unit of time for flows and per iteration for maps,
            or `None` for `'largest'`.

        kaplan_yorke_dimension:
            The Kaplan-Yorke dimension of the spectrum, with `j` the largest number of exponents whose sum is
            nonnegative, or `None` for `'largest'`.

        kolmogorov_sinai_entropy:
            The entropy rate of the dynamics in nats per unit time, the sum of the positive exponents, or `None` for
            `'largest'`.

        values:
            The series of `'largest'`, one per column, or `None` for `'spectrum'`.

        labels:
            The label of every series of `'largest'`, or `None` for `'spectrum'`.

        largest_exponent:
            The largest exponent per unit time of every series of `'largest'`, or `None` for `'spectrum'`.
    """

    problem: ProblemSpec | None = None
    exponents: Array | None = None
    kaplan_yorke_dimension: float | None = None
    kolmogorov_sinai_entropy: float | None = None
    values: Array | None = None
    labels: list[str] | None = None
    largest_exponent: Array | None = None


def lyapunov(
    source: Source,
    *,
    kind: str = 'spectrum',
    variables: Sequence[str] | None = None,
    n_exponents: int | None = None,
    until: float | None = None,
    dt: float | None = None,
    transient: float | None = None,
    parameters: Mapping[str, float] | None = None,
    dimension: int = 3,
    delay: int = 1,
    min_separation: int | None = None,
    trajectory_length: int = 20,
    skip: int = 0,
) -> LyapunovResult:
    """Return the Lyapunov exponents of a system from its tangent dynamics, or the largest exponent of series.

    The kinds read a system or data:

    - `'spectrum'`: the largest `n_exponents` exponents of a flow or a map from the system's initial state. The
      tangent dynamics use the symbolic Jacobian, in which `abs(u)` has the derivative `sign(u)`, and the compiled
      functions take the parameters as an argument, so a sweep over parameter values compiles them once. Tangent
      vectors are re-orthonormalised by QR after every step, as in the Benettin algorithm, and the exponents are the
      average logarithmic growth of the diagonal of `R`. The Kaplan-Yorke dimension and the Kolmogorov-Sinai entropy,
      by Pesin's identity the sum of the positive exponents, follow from the spectrum. An array of exponents passes
      through as the spectrum.
    - `'largest'`: Rosenstein's estimate of the largest exponent of every series, from the divergence of neighbours of
      its delay embedding.

    A positive largest exponent indicates chaos.

    Args:
        source:
            The system, its compiled problem, or a trajectory of it for `'spectrum'`, or an array of exponents, and
            the trajectory or the array of series for `'largest'`.

        kind:
            The exponents to compute, one of the kinds above.

        variables:
            The variables of a trajectory to read for `'largest'`, or `None` for every scalar series.

        n_exponents:
            The number of exponents of the spectrum, or `None` for all of them.

        until:
            The time averaged over, or the number of iterations of a map, or `None` for 1000 for flows and 10,000 for
            maps.

        dt:
            The fixed step of the RK4 integration of flows, or the time between samples of a series, or `None` for
            0.01 and 1.

        transient:
            The time, or the number of iterations, run before averaging, or `None` for a tenth of `until`.

        parameters:
            Values of parameters by dotted path that replace the system's values.

        dimension:
            The dimension of the delay embedding.

        delay:
            The delay of the embedding in steps.

        min_separation:
            The fewest steps between neighbours, or `None` for the mean period from the spectrum.

        trajectory_length:
            The number of steps over which the divergence of neighbours is followed and fitted.

        skip:
            The number of first steps of the divergence left out of the fit, while neighbours align with the unstable
            direction.

    Returns:
        result:
            The spectrum with its dimension and entropy, or the largest exponent of every series.

    Raises:
        AnalysisError:
            The kind is unknown, a setting the kind does not read is changed, or a setting is not positive.

    Examples:
        >>> from skcomplex.core.analysis import lyapunov
        >>> from skcomplex.systems.mathematics.dynamical_systems import logistic_map
        >>> float(lyapunov(logistic_map(r=4.0), until=20000).exponents[0].round(2))
        0.69
        >>> round(lyapunov([0.906, 0.0, -14.572]).kaplan_yorke_dimension, 2)
        2.06
    """
    _check_choice('lyapunov', 'kind', kind, KINDS)
    _check_settings(
        lyapunov,
        'kind',
        SETTINGS[kind],
        kind=kind,
        variables=variables,
        n_exponents=n_exponents,
        until=until,
        dt=dt,
        transient=transient,
        parameters=parameters,
        dimension=dimension,
        delay=delay,
        min_separation=min_separation,
        trajectory_length=trajectory_length,
        skip=skip,
    )
    if (dt is not None and dt <= 0) or dimension < 1 or delay < 1:
        error_msg = f'lyapunov has dt={dt!r}, dimension={dimension!r}, and delay={delay!r}. Use positive values.'
        raise AnalysisError(error_msg)
    if kind == 'largest':
        series = _read_series(source, variables)
        largest = np.asarray(
            [
                nolds.lyap_r(
                    np.asarray(column, dtype=float),
                    emb_dim=dimension,
                    lag=delay,
                    min_tsep=min_separation,
                    tau=dt if dt is not None else 1.0,
                    trajectory_len=trajectory_length,
                    fit='poly',
                    fit_offset=skip,
                )
                for column in series.values.T
            ],
            dtype=float,
        )
        return LyapunovResult(values=series.values, labels=series.labels, largest_exponent=largest)
    problem = None
    if isinstance(source, System | BaseProblemSpec | Trajectory):
        dynamics = StabilityAnalysis._dynamics_of(_read_system(source), parameters)
        problem = dynamics.problem
        exponents = _compute_spectrum(dynamics, n_exponents, until, dt, transient)
    else:
        exponents = np.sort(np.asarray(source, dtype=float).ravel())[::-1]
    return LyapunovResult(
        problem=problem,
        exponents=exponents,
        kaplan_yorke_dimension=_kaplan_yorke(exponents),
        kolmogorov_sinai_entropy=float(exponents[exponents > 0].sum()),
    )


def _compute_spectrum(
    dynamics: _DynamicsSpec,
    n_exponents: int | None,
    until: float | None,
    dt: float | None,
    transient: float | None,
) -> Array:
    """Return the exponents of the system from its tangent dynamics."""
    size = len(dynamics.symbols)
    count = size if n_exponents is None else n_exponents
    problem = dynamics.problem
    indices = {
        name: entry.start + column for entry in problem.states.entries for column, name in enumerate(entry.generic_names)
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
        step = dt if dt is not None else 0.01
        total = until if until is not None else 1000.0
        steps = round(total / step)
        skip = round((transient if transient is not None else total / 10) / step)
        exponents = _flow_spectrum(f, jac, start, values, count, step, steps, skip)
    return np.sort(np.asarray(exponents))[::-1]


def _without_kinks(expression: Expression) -> Expression:
    """Return an expression with each `abs(u)` as `u sgn(u)`, whose derivative is `sgn(u)`, not `u / |u|`."""
    wild = SR.wild(0)
    return SR(expression).subs({abs(wild): wild * sgn(wild)})


def _jacobian(dynamics: _DynamicsSpec) -> list[Expression]:
    """Return the entries of the Jacobian of the rates row by row, with the kinks of absolute values as signs."""
    wild = SR.wild(0)
    rates = [_without_kinks(rate) for rate in dynamics.rates]
    return [rate.diff(item).subs({dirac_delta(wild): 0}) for rate in rates for item in dynamics.symbols]


def _function(
    name: str,
    dynamics: _DynamicsSpec,
    expressions: list[Expression],
    shape: tuple[int, ...],
) -> VectorFunction:
    """Return a Numba function `f(t, x, p)` that fills an array of a shape, compiled once per printed source.

    The parameters are read from `p` in the order of `dynamics.parameters`, so systems of one structure share the
    compiled function whatever their parameter values.
    """
    names = (
        {item: f'x[{index}]' for index, item in enumerate(dynamics.symbols)}
        | {item: f'p[{index}]' for index, item in enumerate(dynamics.parameters)}
        | {dynamics.time: 't'}
    )
    printer = ExpressionPrinter('scalar', names=lambda item: names[item]).prepare()
    printed = printer.run([SR(item) for item in expressions]).code_
    if not isinstance(printed, tuple):
        error_msg = f'The printer returned no code for the Jacobian of {name}.'
        raise AnalysisError(error_msg)
    assignments, codes = printed
    lines = ['import math', 'import numpy as np', '', '']
    lines += [f'def {name}(t, x, p):', f'    out = np.zeros({shape})']
    lines += [f'    {bound} = {code}' for bound, code in assignments]
    lines += [
        f'    out[{np.unravel_index(index, shape) if len(shape) > 1 else index}] = {code}'
        for index, code in enumerate(codes)
    ]
    lines.append('    return out')
    source = '\n'.join(lines) + '\n'
    if source not in FUNCTIONS:
        module = SourceLoader('skcomplex_lyapunov').prepare().run(source).module_
        FUNCTIONS[source] = numba.njit(getattr(module, name))
    return FUNCTIONS[source]


def _kaplan_yorke(exponents: Array) -> float:
    """Return the Kaplan-Yorke dimension `j + Σ_{i<=j} λ_i / |λ_{j+1}|` of a Lyapunov spectrum."""
    ordered = np.sort(np.asarray(exponents, dtype=float))[::-1]
    sums = np.cumsum(ordered)
    nonnegative = np.flatnonzero(sums >= 0)
    if not nonnegative.size:
        return 0.0
    count = int(nonnegative[-1]) + 1
    if count == ordered.size:
        return float(count)
    return float(count + sums[count - 1] / abs(ordered[count]))
