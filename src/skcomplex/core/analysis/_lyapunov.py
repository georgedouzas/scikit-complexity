"""Measure chaos with Lyapunov exponents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import nolds
import numba
import numpy as np
from sage.all__sagemath_symbolics import SR, Expression, dirac_delta, sgn

from ..engine.backends import JaxBackend
from ..engine.compile._pass_compiler import PassCompiler
from ..engine.problems import BaseProblemSpec, ProblemSpec
from ..engine.problems._composite_system import _CompositeSystem
from ..engine.problems._expression_printer import ExpressionPrinter
from ..engine.problems._source_loader import SourceLoader
from ..engine.runtime import Trajectory
from ..engine.types import TracedArray, TracedFunction
from ..enums import DataKind, FormKind
from ..errors import AnalysisError, SkcomplexError
from ..types import Array, VectorFunction
from ._analysis_tags import AnalysisTags
from ._base import _check_choice, _check_settings, _check_tags, _composite, _read_series, _read_system
from ._stability import StabilityResult, _Dynamics
from .types import DataSource, TangentFunction

KINDS = ('spectrum', 'largest')
SETTINGS = {
    'spectrum': {'n_exponents', 'until', 'dt', 'transient', 'parameters'},
    'largest': {'variables', 'dt', 'dimension', 'delay', 'min_separation', 'trajectory_length', 'skip'},
}
FUNCTIONS: dict[str, Callable[..., Array]] = {}
DIFFERENCE_STEP = 1e-6
TAGS = AnalysisTags(DataKind.SYSTEM, randomness='deterministic')
KIND_TAGS = {
    'largest': AnalysisTags(DataKind.SERIES, randomness='deterministic'),
}


@numba.njit
def _flow_spectrum(
    f: VectorFunction,
    tangent: TangentFunction,
    x: Array,
    p: Array,
    n: int,
    dt: float,
    steps: int,
    skip: int,
) -> Array:
    """Integrate a flow and its tangent vectors with RK4, and average the growth.

    The tangent vectors are re-orthonormalised every step, and `tangent(t, x, p, q)` gives the Jacobian times `q`.
    """
    q = np.eye(x.size)[:, :n].copy()
    sums = np.zeros(n)
    t = 0.0
    for step in range(skip + steps):
        k1 = f(t, x, p)
        l1 = tangent(t, x, p, q)
        k2 = f(t + dt / 2, x + dt / 2 * k1, p)
        l2 = tangent(t + dt / 2, x + dt / 2 * k1, p, q + dt / 2 * l1)
        k3 = f(t + dt / 2, x + dt / 2 * k2, p)
        l3 = tangent(t + dt / 2, x + dt / 2 * k2, p, q + dt / 2 * l2)
        k4 = f(t + dt, x + dt * k3, p)
        l4 = tangent(t + dt, x + dt * k3, p, q + dt * l3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        q = q + dt / 6 * (l1 + 2 * l2 + 2 * l3 + l4)
        q, r = np.linalg.qr(q)
        q = np.ascontiguousarray(q)
        if step >= skip:
            sums += np.log(np.abs(np.diag(r)))
        t += dt
    return sums / (steps * dt)


@numba.njit
def _map_spectrum(
    f: VectorFunction,
    tangent: TangentFunction,
    x: Array,
    p: Array,
    n: int,
    steps: int,
    skip: int,
) -> Array:
    """Iterate a map and its tangent vectors, and average the growth.

    The tangent vectors are re-orthonormalised every step, and `tangent(t, x, p, q)` gives the Jacobian times `q`.
    """
    q = np.eye(x.size)[:, :n].copy()
    sums = np.zeros(n)
    for step in range(skip + steps):
        q = tangent(float(step), x, p, q)
        x = f(float(step), x, p)
        q, r = np.linalg.qr(q)
        q = np.ascontiguousarray(q)
        if step >= skip:
            sums += np.log(np.abs(np.diag(r)))
    return sums / steps


@dataclass(frozen=True)
class LyapunovResult:
    """A result of `lyapunov`, the Lyapunov exponents of a system or of series.

    It holds the exponents, the rates at which nearby runs separate along each direction, with the dimension and the
    entropy they give, or the largest exponent of every recorded series, such as `ln 2` for the logistic map at
    `r = 4`. It reports how fast small errors grow in the weather, a double pendulum, and the orbits of asteroids.

    Read more in the user guide, under Analyse.

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

    See Also:
        lyapunov: Returns this result.

    Examples:
        >>> from skcomplex.core.analysis import lyapunov
        >>> round(lyapunov([0.906, 0.0, -14.572]).kaplan_yorke_dimension, 2)
        2.06
    """

    problem: ProblemSpec | None = None
    exponents: Array | None = None
    kaplan_yorke_dimension: float | None = None
    kolmogorov_sinai_entropy: float | None = None
    values: Array | None = None
    labels: list[str] | None = None
    largest_exponent: Array | None = None


def lyapunov(
    source: DataSource,
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
    """Return the Lyapunov exponents of a system, or the largest exponent of series.

    An exponent is the average rate at which two nearby runs separate, `|δx(t)| ≈ |δx(0)| exp(λ t)`, and a positive
    largest exponent means chaos, such as `ln 2` per step for the logistic map at `r = 4`. The exponents of a system
    come from its equations, and the largest exponent of a series from how fast its close returns drift apart. It
    measures how fast small errors grow in the weather, a double pendulum, and the orbits of asteroids.

    Read more in the user guide, under Analyse.

    Args:
        source:
            The system, its compiled problem, or a trajectory of it for `'spectrum'`, or an array of exponents, and
            the trajectory or the array of series for `'largest'`.

        kind:
            What to return. `'spectrum'` gives the largest `n_exponents` exponents of a flow or a map from the initial
            state of the system. The tangent dynamics use the symbolic Jacobian, in which `abs(u)` has the derivative
            `sign(u)`, and the compiled functions take the parameters as an argument, so a sweep over parameter values
            compiles them once. A group of more than 64 members and a field whose laws apply operators, such as a
            spectral Laplacian, have no written out Jacobian, and their tangent vectors follow the forward derivatives
            of JAX when the system has a JAX form, or else central differences of the compiled drift or update, run
            in Python. Tangent vectors are made orthonormal again by QR after every step, as in the algorithm
            of Benettin, and the exponents are the average logarithmic growth of the diagonal of `R`. The Kaplan-Yorke
            dimension and the Kolmogorov-Sinai entropy, by the identity of Pesin the sum of the positive exponents,
            follow from the spectrum, and an array of exponents passes through as the spectrum. `'largest'` gives the
            estimate of Rosenstein of the largest exponent of every series, from the divergence of neighbours of its
            delay embedding.

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

        TagError:
            The system of the source is stochastic, since the exponents need a deterministic system.

    See Also:
        embedding: Gives the points made of delayed copies of one series.
        stability: Finds the fixed points of a system and whether they are stable.

    Examples:
        >>> from skcomplex.core.analysis import lyapunov
        >>> from skcomplex.systems.mathematics.dynamical_systems import LogisticMapSystem
        >>> float(lyapunov(LogisticMapSystem(r=4.0), until=20000).exponents[0].round(2))
        0.69
    """
    source = _composite(source)
    _check_choice('lyapunov', 'kind', kind, KINDS)
    _check_tags(lyapunov, source, kind)
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
    if isinstance(source, _CompositeSystem | BaseProblemSpec | Trajectory):
        problem, values = StabilityResult._get_problem_of(_read_system(source), parameters)
        settings = (n_exponents, until, dt, transient)
        if StabilityResult._is_expandable(problem):
            dynamics = StabilityResult._get_dynamics_of(problem, parameters)
            f, tangent, start, values = _get_symbolic_functions(dynamics)
            exponents = _compute_spectrum(problem, f, tangent, start, values, settings, compiled=True)
        else:
            f, tangent = _get_numeric_functions(problem)
            exponents = _compute_spectrum(problem, f, tangent, problem.initial_state, values, settings, compiled=False)
    else:
        exponents = np.sort(np.asarray(source, dtype=float).ravel())[::-1]
    return LyapunovResult(
        problem=problem,
        exponents=exponents,
        kaplan_yorke_dimension=_kaplan_yorke(exponents),
        kolmogorov_sinai_entropy=float(exponents[exponents > 0].sum()),
    )


def _compute_spectrum(
    problem: ProblemSpec,
    f: VectorFunction,
    tangent: TangentFunction,
    start: Array,
    values: Array,
    settings: tuple[int | None, float | None, float | None, float | None],
    *,
    compiled: bool,
) -> Array:
    """Return the exponents of a system from its tangent dynamics.

    The settings are the number of exponents, the time averaged over, the step, and the transient. The loops run
    compiled by Numba for compiled functions and in Python otherwise.
    """
    n_exponents, until, dt, transient = settings
    count = start.size if n_exponents is None else n_exponents
    x = np.asarray(start, dtype=float)
    if problem.form == FormKind.DISCRETE:
        steps = int(until if until is not None else 10_000)
        skip = int(transient if transient is not None else steps // 10)
        iterate = _map_spectrum if compiled else _map_spectrum.py_func
        exponents = iterate(f, tangent, x, values, count, steps, skip)
    else:
        step = dt if dt is not None else 0.01
        total = until if until is not None else 1000.0
        steps = round(total / step)
        skip = round((transient if transient is not None else total / 10) / step)
        integrate = _flow_spectrum if compiled else _flow_spectrum.py_func
        exponents = integrate(f, tangent, x, values, count, step, steps, skip)
    return np.sort(np.asarray(exponents))[::-1]


def _get_symbolic_functions(dynamics: _Dynamics) -> tuple[VectorFunction, TangentFunction, Array, Array]:
    """Return the compiled rates and tangent dynamics of written out equations.

    It also returns their start and their parameters. The start is the initial state in the order of the equations,
    read from the state vector by component position, and the parameters are the values the compiled functions read,
    in the order of `dynamics.parameters`.
    """
    size = len(dynamics.symbols)
    problem = dynamics.problem
    indices = {
        name: entry.start + index
        for entry in problem.states.entries
        for index, name in enumerate(entry.component_names)
    }
    start = np.asarray([problem.initial_state[indices[name]] for name in dynamics.names], dtype=float)
    f = _function('field', dynamics, dynamics.rates, (size,))
    tangent = _function('tangent', dynamics, _jacobian(dynamics), (size, size))
    values = np.asarray([float(value) for value in dynamics.parameters.values()], dtype=float)
    return f, tangent, start, values


def _get_numeric_functions(problem: ProblemSpec) -> tuple[VectorFunction, TangentFunction]:
    """Return the compiled drift or update of a problem and its tangent dynamics.

    The tangent is the derivative along each tangent vector that JAX gives when the system has a JAX form, and else
    the central difference of the compiled function along each tangent vector.
    """
    f = problem.update if problem.form == FormKind.DISCRETE else problem.drift
    if f is None:
        error_msg = 'The problem has no compiled drift or update to differentiate.'
        raise AnalysisError(error_msg)
    traced = _get_traced(problem)
    if traced is not None:
        return f, _get_jax_tangent(traced)

    def tangent(t: float, x: Array, p: Array, q: Array) -> Array:
        step = DIFFERENCE_STEP * max(1.0, float(np.abs(x).max()))
        columns = [(f(t, x + step * column, p) - f(t, x - step * column, p)) / (2 * step) for column in q.T]
        return np.ascontiguousarray(np.column_stack(columns))

    return f, tangent


def _get_traced(problem: ProblemSpec) -> TracedFunction | None:
    """Return the pure JAX function of the drift or the update of a problem.

    It is `None` when the system has no JAX form.
    """
    if problem.system is None:
        return None
    backend = JaxBackend()
    try:
        compiler = PassCompiler(backend=backend).prepare(problem.system)
        compiler.run()
    except SkcomplexError:
        return None
    if any(issue.code == 'not-differentiable' for issue in compiler.report_.issues):
        return None
    return backend.traced_


def _get_jax_tangent(traced: TracedFunction) -> TangentFunction:
    """Return the tangent dynamics from the forward derivatives of a pure JAX function.

    They run in double precision.
    """

    def push(t: TracedArray, x: TracedArray, p: TracedArray, q: TracedArray) -> TracedArray:
        along = jax.vmap(lambda vector: jax.jvp(lambda y: traced(t, y, p), (x,), (vector,))[1], in_axes=1, out_axes=1)
        return along(q)

    compiled = jax.jit(push)

    def tangent(t: float, x: Array, p: Array, q: Array) -> Array:
        with jax.enable_x64(True):
            pushed = compiled(jnp.asarray(t), jnp.asarray(x), jnp.asarray(p), jnp.asarray(q))
            return np.ascontiguousarray(np.asarray(pushed, dtype=float))

    return tangent


def _without_kinks(expression: Expression) -> Expression:
    """Return an expression with each `abs(u)` as `u sgn(u)`.

    Its derivative is then `sgn(u)`, not `u / |u|`.
    """
    wild = SR.wild(0)
    return SR(expression).subs({abs(wild): wild * sgn(wild)})


def _jacobian(dynamics: _Dynamics) -> list[Expression]:
    """Return the entries of the Jacobian of the rates row by row.

    The kinks of absolute values become signs.
    """
    wild = SR.wild(0)
    rates = [_without_kinks(rate) for rate in dynamics.rates]
    return [rate.diff(item).subs({dirac_delta(wild): 0}) for rate in rates for item in dynamics.symbols]


def _function(
    name: str,
    dynamics: _Dynamics,
    expressions: list[Expression],
    shape: tuple[int, ...],
) -> Callable[..., Array]:
    """Return a Numba function `f(t, x, p)` that fills an array of a shape.

    A square array is the Jacobian, and its function `f(t, x, p, q)` returns the Jacobian times `q`. It is compiled
    once per printed source. The parameters are read from `p` in the order of `dynamics.parameters`, so
    systems of one structure share the compiled function whatever their parameter values.
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
    arguments = 't, x, p, q' if len(shape) > 1 else 't, x, p'
    lines += [f'def {name}({arguments}):', f'    out = np.zeros({shape})']
    lines += [f'    {bound} = {code}' for bound, code in assignments]
    lines += [
        f'    out[{np.unravel_index(index, shape) if len(shape) > 1 else index}] = {code}'
        for index, code in enumerate(codes)
    ]
    lines.append('    return out @ q' if len(shape) > 1 else '    return out')
    source = '\n'.join(lines) + '\n'
    if source not in FUNCTIONS:
        module = SourceLoader('skcomplex_lyapunov').prepare().run(source).module_
        FUNCTIONS[source] = numba.njit(getattr(module, name))
    return FUNCTIONS[source]


def _kaplan_yorke(exponents: Array) -> float:
    """Return the Kaplan-Yorke dimension of a Lyapunov spectrum.

    It is `j + Σ_{i<=j} λ_i / |λ_{j+1}|`.
    """
    ordered = np.sort(np.asarray(exponents, dtype=float))[::-1]
    sums = np.cumsum(ordered)
    nonnegative = np.flatnonzero(sums >= 0)
    if not nonnegative.size:
        return 0.0
    count = int(nonnegative[-1]) + 1
    if count == ordered.size:
        return float(count)
    return float(count + sums[count - 1] / abs(ordered[count]))
