"""Measure chaos with Lyapunov exponents: the spectrum of a system from its tangent dynamics, or the largest of data."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence

import nolds
import numba
import numpy as np
from sage.all__sagemath_symbolics import SR, Expression, dirac_delta, sgn

from ..components.system import System
from ..engine.problems import BaseProblemSpec, ExpressionPrinter, ProblemSpec, SourceLoader
from ..engine.problems._dynamics_spec import _DynamicsSpec
from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..types import Array, DataKind, VectorFunction
from ._base import BaseAnalysis
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


class LyapunovAnalysis(BaseAnalysis):
    """Lyapunov exponents of a system from its tangent dynamics, or the largest exponent of recorded series.

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

    Attributes:
        problem_:
            The compiled problem of the spectrum of a system, or `None` for an array of exponents.

        exponents_:
            The exponents of the spectrum in decreasing order, per unit of time for flows and per iteration for maps.

        kaplan_yorke_dimension_:
            The Kaplan-Yorke dimension of the spectrum, with `j` the largest number of exponents whose sum is
            nonnegative.

        kolmogorov_sinai_entropy_:
            The entropy rate of the dynamics in nats per unit time, the sum of the positive exponents.

        values_:
            The series of `'largest'`, one per column.

        labels_:
            The label of every series of `'largest'`.

        largest_exponent_:
            The largest exponent per unit time of every series of `'largest'`.

    Examples:
        >>> from skcomplex.core.analysis import LyapunovAnalysis
        >>> from skcomplex.systems.mathematics.dynamical_systems import logistic_map
        >>> analysis = LyapunovAnalysis(until=20000).prepare(logistic_map(r=4.0)).run()
        >>> float(analysis.exponents_[0].round(2))
        0.69
        >>> round(LyapunovAnalysis().prepare([0.906, 0.0, -14.572]).run().kaplan_yorke_dimension_, 2)
        2.06
    """

    def __init__(
        self,
        kind: str = 'spectrum',
        variables: Sequence[str] | None = None,
        *,
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
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.n_exponents = n_exponents
        self.until = until
        self.dt = dt
        self.transient = transient
        self.parameters = parameters
        self.dimension = dimension
        self.delay = delay
        self.min_separation = min_separation
        self.trajectory_length = trajectory_length
        self.skip = skip

    def _check(self) -> None:
        """Validate the kind, that the kind reads every changed setting, and the positive settings."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_settings('kind', SETTINGS[self.kind])
        if (self.dt is not None and self.dt <= 0) or self.dimension < 1 or self.delay < 1:
            error_msg = (
                f'{type(self).__name__} has dt={self.dt!r}, dimension={self.dimension!r}, and delay={self.delay!r}. '
                'Use positive values.'
            )
            raise AnalysisError(error_msg)

    def _get_data_kind(self) -> DataKind:
        """Return the system kind for the spectrum, and the series kind for the largest exponent of data."""
        return DataKind.SYSTEM if self.kind == 'spectrum' else DataKind.SERIES

    def _prepare(self, source: Source) -> None:
        """Read the series for the largest exponent, or the dynamics of the system or the exponents for the spectrum."""
        if self.kind == 'largest':
            series = self._read_series(source, self.variables)
            self.values_ = series.values
            self.labels_ = series.labels
            return
        if isinstance(source, System | BaseProblemSpec | Trajectory):
            dynamics = StabilityAnalysis._dynamics_of(self._read_system(source), self.parameters)
            self.problem_: ProblemSpec | None = dynamics.problem
            return
        self.problem_ = None
        self.exponents_ = np.sort(np.asarray(source, dtype=float).ravel())[::-1]

    def _compute_spectrum(self, dynamics: _DynamicsSpec) -> Array:
        """Return the exponents of the system from its tangent dynamics."""
        size = len(dynamics.symbols)
        count = size if self.n_exponents is None else self.n_exponents
        problem = dynamics.problem
        indices = {
            name: entry.start + column
            for entry in problem.states.entries
            for column, name in enumerate(entry.generic_names)
        }
        start = np.asarray([problem.initial_state[indices[name]] for name in dynamics.names], dtype=float)
        f = self._function('field', dynamics, dynamics.rates, (size,))
        jac = self._function('jacobian', dynamics, self._jacobian(dynamics), (size, size))
        values = np.asarray([float(value) for value in dynamics.parameters.values()], dtype=float)
        if dynamics.discrete:
            steps = int(self.until if self.until is not None else 10_000)
            skip = int(self.transient if self.transient is not None else steps // 10)
            exponents = _map_spectrum(f, jac, start, values, count, steps, skip)
        else:
            dt = self.dt if self.dt is not None else 0.01
            total = self.until if self.until is not None else 1000.0
            steps = round(total / dt)
            skip = round((self.transient if self.transient is not None else total / 10) / dt)
            exponents = _flow_spectrum(f, jac, start, values, count, dt, steps, skip)
        return np.sort(np.asarray(exponents))[::-1]

    def _run(self, inputs: object = None) -> None:
        """Compute the spectrum with its dimension and entropy, or the largest exponent of every series."""
        if self.kind == 'largest':
            self.largest_exponent_ = np.asarray(
                [
                    nolds.lyap_r(
                        np.asarray(column, dtype=float),
                        emb_dim=self.dimension,
                        lag=self.delay,
                        min_tsep=self.min_separation,
                        tau=self.dt if self.dt is not None else 1.0,
                        trajectory_len=self.trajectory_length,
                        fit='poly',
                        fit_offset=self.skip,
                    )
                    for column in self.values_.T
                ],
                dtype=float,
            )
            return
        if self.problem_ is not None:
            self.exponents_ = self._compute_spectrum(StabilityAnalysis._dynamics_of(self.problem_, self.parameters))
        self.kaplan_yorke_dimension_ = self._kaplan_yorke(self.exponents_)
        self.kolmogorov_sinai_entropy_ = float(self.exponents_[self.exponents_ > 0].sum())

    @staticmethod
    def _without_kinks(expression: Expression) -> Expression:
        """Return an expression with each `abs(u)` as `u sgn(u)`, whose derivative is `sgn(u)`, not `u / |u|`."""
        wild = SR.wild(0)
        return SR(expression).subs({abs(wild): wild * sgn(wild)})

    @staticmethod
    def _jacobian(dynamics: _DynamicsSpec) -> list[Expression]:
        """Return the entries of the Jacobian of the rates row by row, with the kinks of absolute values as signs."""
        wild = SR.wild(0)
        rates = [LyapunovAnalysis._without_kinks(rate) for rate in dynamics.rates]
        return [rate.diff(item).subs({dirac_delta(wild): 0}) for rate in rates for item in dynamics.symbols]

    @staticmethod
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

    @staticmethod
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
