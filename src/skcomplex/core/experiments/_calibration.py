"""Fit the model parameters of a system to observed time series."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import logging
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import numpyro
import numpyro.distributions as dist
import optax
import optuna
import pyabc
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from numpyro.infer import AIES, MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal
from scipy import stats
from scipy.optimize import least_squares, minimize

from ..components.systems import BaseSystem
from ..engine.backends import JaxBackend
from ..engine.compile._pass_compiler import PassCompiler
from ..engine.problems import ProblemSpec
from ..engine.runtime import Trajectory, simulate
from ..engine.solvers import BaseSolver
from ..engine.types import TracedArray, TracedFunction
from ..enums import CalibrationMethodKind, FormKind, LossKind
from ..errors import SkcomplexError, ValidationError
from ..types import Array
from ._designs import _check_bounds
from ._runs import _check_count, _check_source
from ._utils import build_system
from .types import Bounds, Source

LOSSES: dict[str, tuple[LossKind, ...]] = {
    CalibrationMethodKind.LOCAL: (LossKind.SQUARES, LossKind.LIKELIHOOD, LossKind.EVENTS),
    CalibrationMethodKind.GLOBAL: (LossKind.SQUARES, LossKind.LIKELIHOOD, LossKind.EVENTS),
    CalibrationMethodKind.ABC_SMC: (LossKind.SQUARES,),
    CalibrationMethodKind.MCMC: (LossKind.LIKELIHOOD, LossKind.EVENTS),
    CalibrationMethodKind.VARIATIONAL: (LossKind.LIKELIHOOD, LossKind.SQUARES),
}
BUDGETS: dict[str, int] = {
    CalibrationMethodKind.LOCAL: 0,
    CalibrationMethodKind.GLOBAL: 200,
    CalibrationMethodKind.ABC_SMC: 20000,
    CalibrationMethodKind.MCMC: 0,
    CalibrationMethodKind.VARIATIONAL: 2000,
}
CENTRAL = (0.0005, 0.9995)
PRIOR_FAMILIES = ('uniform', 'norm', 'lognorm', 'truncnorm', 'beta', 'gamma', 'expon')
PREDICTIVE_DRAWS = 200
PREDICTIVE_QUANTILES = (0.05, 0.5, 0.95)
RHAT_LIMIT = 1.05
LEARNING_RATE = 0.01
SEED_LIMIT = 2**32


type Series = dict[str, tuple[Array, Array]]
type Summary = Callable[[Mapping[str, Array]], npt.ArrayLike]


@dataclass(frozen=True)
class CalibrationResult:
    """A result of `calibrate`, the fitted parameters and system, with diagnostics.

    A point fit holds one estimate. A posterior method also holds the samples, their weights, the diagnostics of the
    method, and the posterior predictive check, and its estimates are the posterior medians.

    Args:
        system:
            The system with the fitted model parameters.

        estimates:
            The fitted value of every parameter, the posterior median for the posterior methods.

        r2:
            The coefficient of determination over all data points.

        rmse:
            The root mean squared error.

        mae:
            The mean absolute error.

        mape:
            The mean absolute percentage error, over data points that are not zero.

        covariance:
            The covariance of the estimates, from the Jacobian of the residuals for a point fit and from the weighted
            samples for a posterior method, or `None` when it is singular or there are no residuals.

        simulated:
            The simulated values at the data times, by path.

        success:
            Whether the method converged, such as the optimiser of a point fit, the split R-hat of every parameter of
            MCMC, or a finite loss of variational calibration.

        samples:
            The samples of every parameter of a posterior method, or `None` for a point fit.

        weights:
            The normalised weights of the samples, or `None` for a point fit.

        diagnostics:
            The diagnostics of the method, such as the tolerances and acceptance rates of ABC-SMC, the split R-hat and
            effective sample sizes of MCMC, or the loss history of variational calibration.

        predictive:
            The 5%, 50%, and 95% quantiles of the simulated series of posterior draws at the data times by path, under
            `q05`, `q50`, and `q95`, and the share of observed points inside the band under `coverage`, or `None` for
            a point fit.
    """

    system: BaseSystem
    estimates: dict[str, float]
    r2: float
    rmse: float
    mae: float
    mape: float
    covariance: Array | None
    simulated: dict[str, Array]
    success: bool
    samples: dict[str, Array] | None = None
    weights: Array | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)
    predictive: dict[str, object] | None = None


class _CalibrationObjective:
    """Objective of a calibration, which every method searches or samples.

    It builds the candidate system of a parameter vector, simulates it at the data times, and scores it by residuals,
    the Gaussian log-likelihood with an unknown variance, the log-probability of recorded draws, and the log prior. It
    also gives the summaries that ABC-SMC compares.

    Args:
        source:
            The system or the factory of systems.

        series:
            The observed `(times, values)` by path.

        priors:
            The prior of every parameter, a frozen SciPy distribution.

        replicates:
            The number of runs averaged per evaluation.

        dt:
            The time between frames, or `None` for the tuned one.

        solver:
            The solver, or `None` for the tuned one.

        random_state:
            The seed of the runs.

        recorded:
            The trajectory recorded with its draws that the event likelihood replays, or `None`.

        summary:
            The function from series by path to summary statistics, or `None` for the concatenated series.
    """

    def __init__(
        self,
        source: Source,
        series: Series,
        priors: Mapping[str, stats.rv_continuous],
        *,
        replicates: int = 1,
        dt: float | None = None,
        solver: BaseSolver | None = None,
        random_state: int | None = None,
        recorded: Trajectory | None = None,
        summary: Summary | None = None,
    ) -> None:
        self.source = source
        self.series = series
        self.priors = priors
        self.replicates = replicates
        self.dt = dt
        self.solver = solver
        self.random_state = random_state
        self.recorded = recorded
        self.summary = summary

    @property
    def names(self) -> list[str]:
        """Names of the parameters, in the order of parameter vectors."""
        return list(self.priors)

    @property
    def observed(self) -> Array:
        """Observed values of every path, concatenated."""
        return np.concatenate([values for _, values in self.series.values()])

    @property
    def box(self) -> tuple[Array, Array]:
        """Lower and upper bounds of point fits.

        They are the pair of a uniform prior and the central 99.9% of any other prior.
        """
        quantiles = [(0.0, 1.0) if prior.dist.name == 'uniform' else CENTRAL for prior in self.priors.values()]
        low = np.asarray([prior.ppf(lower) for prior, (lower, _) in zip(self.priors.values(), quantiles, strict=True)])
        high = np.asarray([prior.ppf(upper) for prior, (_, upper) in zip(self.priors.values(), quantiles, strict=True)])
        return low.astype(float), high.astype(float)

    @property
    def seeds(self) -> list[np.random.SeedSequence]:
        """Fixed streams of the replicates, so the loss is smooth in the parameters."""
        return np.random.SeedSequence(self.random_state).spawn(self.replicates)

    def build(self, values: Sequence[float]) -> BaseSystem:
        """Return the candidate system of a parameter vector."""
        return build_system(self.source, dict(zip(self.names, map(float, values), strict=True)))

    def simulate(
        self,
        values: Sequence[float],
        seeds: Sequence[np.random.SeedSequence] | None = None,
    ) -> dict[str, Array]:
        """Return the simulated values at the data times by path, averaged over runs.

        A calibration on recorded draws replays them, and the others run once per seed, by default the fixed streams of
        the replicates.
        """
        candidate = self.build(values)
        until = max(float(times.max()) for times, _ in self.series.values())
        runs = []
        for seed in self.seeds if seeds is None else seeds:
            trajectory = simulate(
                candidate,
                until=until,
                dt=self.dt,
                solver=self.solver,
                random_state=np.random.default_rng(seed),
            )
            runs.append(self._get_at_times(trajectory))
        return {path: np.mean([run[path] for run in runs], axis=0) for path in self.series}

    def _get_at_times(self, trajectory: Trajectory) -> dict[str, Array]:
        """Return the values of a trajectory interpolated at the data times by path."""
        return {
            path: np.interp(times, trajectory.times, np.asarray(trajectory.get_values(path), dtype=float))
            for path, (times, _) in self.series.items()
        }

    def _replay(self, values: Sequence[float]) -> Trajectory:
        """Return the replay of the recorded draws under a parameter vector."""
        recorded = self.recorded
        if recorded is None:
            error_msg = 'The event likelihood needs a trajectory recorded with its draws.'
            raise ValidationError(error_msg)
        return simulate(
            self.build(values),
            until=float(recorded.times[-1]),
            dt=self.dt,
            solver=self.solver,
            random_state=self.random_state,
            replay=recorded,
        )

    def get_replayed(self, values: Sequence[float]) -> dict[str, Array]:
        """Return the replayed values at the data times by path."""
        return self._get_at_times(self._replay(values))

    def get_residuals(self, values: Sequence[float]) -> Array:
        """Return the simulated minus the observed values of every path, concatenated."""
        simulated = self.simulate(values)
        return np.concatenate([simulated[path] - observed for path, (_, observed) in self.series.items()])

    def log_likelihood(self, values: Sequence[float]) -> float:
        """Return the Gaussian log-likelihood with the variance profiled out.

        It is exact up to a constant.
        """
        residuals = self.get_residuals(values)
        return float(-0.5 * residuals.size * np.log(np.sum(residuals**2)))

    def get_event_log_likelihood(self, values: Sequence[float]) -> float:
        """Return the log-probability of the recorded draws replayed under parameters.

        A replay whose draws do not match the recording, such as a group that grows otherwise, has probability 0.
        """
        try:
            probability = self._replay(values).log_probability
        except ValidationError:
            return -np.inf
        return -np.inf if probability is None else float(probability)

    def get_loss(self, values: Sequence[float], loss: str) -> float:
        """Return the loss of a parameter vector to minimise.

        It is half the sum of squares, the negative Gaussian log-likelihood, or the negative event log-likelihood.
        """
        if loss == 'squares':
            return float(0.5 * np.sum(self.get_residuals(values) ** 2))
        if loss == 'likelihood':
            return -self.log_likelihood(values)
        return -self.get_event_log_likelihood(values)

    def get_summaries(self, simulated: Mapping[str, Array]) -> Array:
        """Return the summary statistics of series by path."""
        if self.summary is None:
            return np.concatenate([np.asarray(simulated[path], dtype=float) for path in self.series])
        return np.atleast_1d(np.asarray(self.summary(simulated), dtype=float))

    def _get_compiled(self, values: Sequence[float]) -> tuple[ProblemSpec, TracedFunction]:
        """Return the JAX problem of the candidate of a parameter vector, and its function.

        Args:
            values:
                The parameter vector.

        Returns:
            compiled:
                The problem and the pure JAX function of its form.

        Raises:
            ValidationError:
                The candidate has no JAX form, draws a discrete draw without a temperature, or is neither a discrete
                map nor ordinary differential equations.
        """
        backend = JaxBackend()
        try:
            compiler = PassCompiler(backend=backend).prepare(self.build(values).prepare().system_)
            problem = compiler.run().problem_
        except SkcomplexError as error:
            error_msg = f'The system has no differentiable run on JAX: {error}'
            raise ValidationError(error_msg) from error
        stops = [issue.text for issue in compiler.report_.issues if issue.code == 'not-differentiable']
        if stops:
            error_msg = f'The system has no differentiable run on JAX: {" ".join(stops)}'
            raise ValidationError(error_msg)
        if not isinstance(problem, ProblemSpec) or problem.form not in (FormKind.ODE, FormKind.DISCRETE):
            error_msg = f'Differentiable runs unroll maps and ordinary differential equations, not a {problem.form}.'
            raise ValidationError(error_msg)
        return problem, backend.traced_

    def get_traced(self) -> Callable[..., TracedArray]:
        """Return the JAX function from a parameter vector to the simulated data values.

        The values at the data times are concatenated.

        The function `predict(theta, key=None)` takes a JAX key when the system draws, which draws its uniform and
        normal values.

        The system is compiled on JAX at the medians of the priors and at small steps of every parameter, and the
        initial states and the parameter vector of its problem must be affine in the model parameters, as they are
        when model parameters set values of components.

        Returns:
            predict:
                The function from a parameter vector, and a JAX key for a system that draws, to the simulated values.

        Raises:
            ValidationError:
                The problem changes its structure or is not affine in a model parameter, or a path is not a state.
        """
        center = np.asarray([float(prior.median()) for prior in self.priors.values()])
        problem, function = self._get_compiled(center)
        columns = []
        for index, name in enumerate(self.names):
            step = 1e-3 * max(1.0, abs(center[index]))
            slopes = []
            for factor in (1.0, 2.0):
                shifted = center.copy()
                shifted[index] += factor * step
                other = self._get_compiled(shifted)[0]
                if other.initial_state.size != problem.initial_state.size or (
                    other.initial_parameters.size != problem.initial_parameters.size
                ):
                    error_msg = f'The model parameter {name} changes the structure of the system.'
                    raise ValidationError(error_msg)
                changed = np.concatenate([other.initial_state - problem.initial_state, other.initial_parameters])
                base = np.concatenate([np.zeros(problem.initial_state.size), problem.initial_parameters])
                slopes.append((changed - base) / (factor * step))
            if not np.allclose(slopes[0], slopes[1], rtol=1e-6, atol=1e-9):
                error_msg = f'Differentiable runs need values of components that are affine in {name}.'
                raise ValidationError(error_msg)
            columns.append(slopes[0])
        jacobian = np.stack(columns, axis=1)
        start = np.concatenate([problem.initial_state, problem.initial_parameters])
        size = problem.initial_state.size
        times = np.unique(np.concatenate([times for times, _ in self.series.values()]))
        run = JaxBackend.unroll(function, problem.form, times, throw=False, draws=problem.draws)
        reads = []
        for path, (path_times, _) in self.series.items():
            if not problem.states.has(path):
                error_msg = f'Differentiable runs read states, and {path} is not a state of the system.'
                raise ValidationError(error_msg)
            indices = np.arange(problem.states.size)[problem.states.locate(path)[1]]
            reads.append((np.searchsorted(times, path_times), indices))

        def predict(theta: TracedArray, key: TracedArray | None = None) -> TracedArray:
            vector = jnp.asarray(start) + jnp.asarray(jacobian) @ (theta - jnp.asarray(center))
            states = run(vector[:size], vector[size:], key)
            return jnp.concatenate([states[rows][:, indices].ravel() for rows, indices in reads])

        return predict

    def get_predictive(
        self,
        samples: Array,
        weights: Array,
        loss: str,
        rng: np.random.Generator,
    ) -> dict[str, object]:
        """Return the quantiles of simulated series of posterior draws, and their coverage.

        The quantiles are taken at the data times.

        At most 200 draws are chosen by the weights and run with fresh streams. For the Gaussian likelihood every draw
        also adds observation noise of the variance its residuals give.
        """
        if not self.series:
            empty: dict[str, Array] = {}
            return {'q05': empty, 'q50': empty, 'q95': empty, 'coverage': float('nan')}
        chosen = rng.choice(len(samples), size=min(len(samples), PREDICTIVE_DRAWS), p=weights)
        observed = self.observed
        runs = []
        for row in samples[chosen]:
            seed = np.random.SeedSequence(int(rng.integers(SEED_LIMIT)))
            simulated = self.simulate(row, [seed])
            values = np.concatenate([simulated[path] for path in self.series])
            if loss == 'likelihood':
                values = values + rng.normal(0.0, np.sqrt(np.mean((values - observed) ** 2)), values.size)
            runs.append(values)
        quantiles = np.quantile(np.asarray(runs), PREDICTIVE_QUANTILES, axis=0)
        cuts = np.cumsum([0, *(values.size for _, values in self.series.values())])
        bands = {
            key: {path: quantiles[row, cuts[index] : cuts[index + 1]] for index, path in enumerate(self.series)}
            for row, key in enumerate(('q05', 'q50', 'q95'))
        }
        inside = (observed >= quantiles[0]) & (observed <= quantiles[2])
        return {**bands, 'coverage': float(np.mean(inside))}

    def get_covariance(self, estimate: Array, loss: str) -> Array | None:
        """Return the covariance of point estimates from the Jacobian of the residuals.

        It is `None` for the event likelihood, which has no residuals, and when the Jacobian is singular.
        """
        if loss == 'events':
            return None
        low, high = self.box
        fit = least_squares(self.get_residuals, np.clip(estimate, low, high), bounds=(low, high), max_nfev=1)
        jacobian, final = fit.jac, self.get_residuals(estimate)
        dof = max(final.size - len(self.names), 1)
        try:
            return np.asarray(np.linalg.inv(jacobian.T @ jacobian) * float(np.sum(final**2)) / dof)
        except np.linalg.LinAlgError:
            return None


def _check_priors(bounds: Bounds) -> dict[str, stats.rv_continuous]:
    """Return the prior of every parameter from its bounds.

    A pair `(low, high)` is the uniform prior on it, and a SciPy distribution is the prior itself.

    Args:
        bounds:
            The pair or the SciPy distribution of every parameter by name.

    Returns:
        priors:
            The SciPy distribution of every parameter by name.

    Raises:
        ValidationError:
            A bound is neither a pair nor a SciPy distribution, or a pair is not increasing.
    """
    priors = {}
    for name, bound in _check_bounds(bounds).items():
        if isinstance(bound, Sequence):
            low, high = (float(item) for item in bound)
            if not high > low:
                error_msg = f'The bounds of {name} must be a pair (low, high) with low < high, not {bound!r}.'
                raise ValidationError(error_msg)
            priors[name] = stats.uniform(low, high - low)
        else:
            priors[name] = bound
    return priors


def _check_settings(method: str, loss: str, data: object, bounds: Bounds, replicates: int) -> None:
    """Raise an error for an invalid method, loss, data, replicates, or bounds."""
    if method not in CalibrationMethodKind:
        error_msg = f'Unknown method {method!r}. Use one of {", ".join(CalibrationMethodKind)}.'
        raise ValidationError(error_msg)
    if loss not in LOSSES[method]:
        start = f'Unknown loss {loss!r}.' if loss not in LossKind else f'The method {method!r} does not take {loss!r}.'
        error_msg = f'{start} The method {method!r} takes the losses {", ".join(LOSSES[method])}.'
        raise ValidationError(error_msg)
    if loss == 'events' and not (isinstance(data, Trajectory) and data.draws is not None):
        error_msg = 'The loss events takes a trajectory recorded with `simulate(..., record_draws=True)` as data.'
        raise ValidationError(error_msg)
    if loss != 'events' and not data:
        error_msg = 'Calibration needs at least one observed series.'
        raise ValidationError(error_msg)
    _check_count('replicates', replicates)
    dotted = sorted(name for name in bounds if '.' in name)
    if dotted:
        error_msg = (
            f'Calibration fits model parameters of the system, which `set_params` takes, not the dotted paths '
            f'{", ".join(dotted)}.'
        )
        raise ValidationError(error_msg)


def _series_of(data: Mapping[str, tuple[npt.ArrayLike, npt.ArrayLike]] | Trajectory) -> Series:
    """Return the observed series by path.

    A recorded trajectory gives its scalar states at its frames.
    """
    if isinstance(data, Trajectory):
        states = data.problem.states
        return {
            name: (np.asarray(data.times, dtype=float), np.asarray(data.get_values(name), dtype=float))
            for name in states.names
            if states.get_entry(name).size == 1
        }
    return {
        path: (np.asarray(times, dtype=float), np.asarray(values, dtype=float))
        for path, (times, values) in data.items()
    }


def _fit_local(objective: _CalibrationObjective, loss: str, start: Array) -> tuple[Array, bool]:
    """Return the local SciPy fit of a loss from a start, and whether it converged."""
    low, high = objective.box
    if loss == 'squares':
        fit = least_squares(objective.get_residuals, start, bounds=(low, high), x_scale='jac')
        return np.asarray(fit.x, dtype=float), bool(fit.success)
    if loss == 'likelihood':
        count = objective.observed.size
        fit = minimize(
            lambda values: 0.5 * count * np.log(np.sum(objective.get_residuals(values) ** 2)),
            start,
            bounds=list(zip(low, high, strict=True)),
        )
    else:
        fit = minimize(
            lambda values: min(objective.get_loss(values, loss), np.finfo(float).max),
            start,
            bounds=list(zip(low, high, strict=True)),
            method='Nelder-Mead',
        )
    return np.asarray(fit.x, dtype=float), bool(fit.success)


def _fit_global(objective: _CalibrationObjective, loss: str, budget: int, random_state: int | None) -> Array:
    """Return the best of `budget` evaluations of a loss chosen by a Parzen estimator.

    The estimator is a seeded tree-structured Parzen estimator.
    """
    low, high = objective.box
    names = objective.names

    def evaluate(trial: optuna.Trial) -> float:
        values = [trial.suggest_float(name, lower, upper) for name, lower, upper in zip(names, low, high, strict=True)]
        return min(objective.get_loss(values, loss), np.finfo(float).max)

    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(logging.WARNING)
    try:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(evaluate, n_trials=budget)
    finally:
        optuna.logging.set_verbosity(verbosity)
    return np.asarray([study.best_params[name] for name in names], dtype=float)


def _numpyro_prior(name: str, prior: stats.rv_continuous) -> dist.Distribution:
    """Return the numpyro distribution of a SciPy prior.

    The families are uniform, normal, and truncated normal with any location and scale, log-normal, gamma, and
    exponential with a scale, and the standard beta distribution.

    Args:
        name:
            The name of the parameter.

        prior:
            The SciPy distribution of its prior.

    Returns:
        distribution:
            The numpyro distribution.

    Raises:
        ValidationError:
            The family has no numpyro distribution here, or it has a location or scale numpyro does not take.
    """
    family = prior.dist.name
    if family not in PRIOR_FAMILIES:
        error_msg = (
            f'The prior of {name} is a SciPy {family} distribution, which MCMC and variational calibration do not '
            f'take. Use one of {", ".join(PRIOR_FAMILIES)}, or a pair (low, high).'
        )
        raise ValidationError(error_msg)
    names = [item.strip() for item in (prior.dist.shapes or '').split(',') if item.strip()]
    arguments = list(prior.args)
    shapes = [*arguments[: len(names)], *(prior.kwds[item] for item in names[len(arguments) :])]
    loc = arguments[len(names)] if len(arguments) > len(names) else prior.kwds.get('loc', 0.0)
    scale = arguments[len(names) + 1] if len(arguments) > len(names) + 1 else prior.kwds.get('scale', 1.0)
    loc, scale = float(loc), float(scale)
    if family in {'lognorm', 'gamma', 'expon', 'beta'} and loc != 0:
        error_msg = f'The prior of {name} is a SciPy {family} distribution with a location, which numpyro lacks.'
        raise ValidationError(error_msg)
    if family == 'beta' and scale != 1:
        error_msg = f'The prior of {name} is a SciPy beta distribution with a scale, which numpyro does not take.'
        raise ValidationError(error_msg)
    distributions: dict[str, Callable[[], dist.Distribution]] = {
        'uniform': lambda: dist.Uniform(loc, loc + scale),
        'norm': lambda: dist.Normal(loc, scale),
        'lognorm': lambda: dist.LogNormal(np.log(scale), shapes[0]),
        'truncnorm': lambda: dist.TruncatedNormal(
            loc,
            scale,
            low=loc + shapes[0] * scale,
            high=loc + shapes[1] * scale,
        ),
        'beta': lambda: dist.Beta(shapes[0], shapes[1]),
        'gamma': lambda: dist.Gamma(shapes[0], 1 / scale),
        'expon': lambda: dist.Exponential(1 / scale),
    }
    return distributions[family]()


def _weighted_median(values: Array, weights: Array) -> float:
    """Return the weighted median of values."""
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def _fit_abc(
    objective: _CalibrationObjective,
    n_samples: int,
    budget: int,
    random_state: int | None,
) -> tuple[Array, Array, dict[str, object]]:
    """Return the weighted samples of ABC-SMC with pyabc and its generations.

    Each generation gives its tolerance and acceptance.

    pyabc draws from the global NumPy random state, which is seeded from the random state and restored after, and its
    history lives in a temporary SQLite file.
    """
    names = objective.names
    priors = pyabc.Distribution(
        **{name: pyabc.RV(prior.dist.name, *prior.args, **prior.kwds) for name, prior in objective.priors.items()},
    )
    rng = np.random.default_rng(random_state)

    def model(parameters: Mapping[str, float]) -> dict[str, float]:
        seed = np.random.SeedSequence(int(rng.integers(SEED_LIMIT)))
        summaries = objective.get_summaries(objective.simulate([parameters[name] for name in names], [seed]))
        return {f's{index}': float(value) for index, value in enumerate(summaries)}

    observed = objective.get_summaries({path: values for path, (_, values) in objective.series.items()})
    abc = pyabc.ABCSMC(
        model,
        priors,
        pyabc.AdaptivePNormDistance(p=2),
        population_size=n_samples,
        sampler=pyabc.sampler.SingleCoreSampler(),
    )
    state = np.random.get_state()
    logger = logging.getLogger('ABC')
    level = logger.level
    np.random.seed(int(rng.integers(SEED_LIMIT)))
    logger.setLevel(logging.WARNING)
    try:
        with tempfile.TemporaryDirectory() as directory:
            database = f'sqlite:///{Path(directory) / "abc.db"}'
            abc.new(database, {f's{index}': float(value) for index, value in enumerate(observed)})
            history = abc.run(minimum_epsilon=0.0, max_total_nr_simulations=budget)
            frame, weights = history.get_distribution(m=0, t=history.max_t)
            populations = history.get_all_populations()
    finally:
        np.random.set_state(state)
        logger.setLevel(level)
    populations = populations[populations['t'] >= 0]
    diagnostics: dict[str, object] = {
        'epsilons': np.asarray(populations['epsilon'], dtype=float),
        'acceptance_rates': np.asarray(populations['particles'] / populations['samples'], dtype=float),
        'simulations': int(populations['samples'].sum()),
        'summaries': observed.size,
    }
    samples = np.asarray(frame[names], dtype=float)
    return samples, np.asarray(weights, dtype=float) / np.sum(weights), diagnostics


def _chain_diagnostics(draws: Mapping[str, Array], names: list[str]) -> tuple[Array, dict[str, object], bool]:
    """Return the samples of chains by row, their R-hat, sample sizes, and convergence.

    The R-hat is split and the sample sizes are effective.
    """
    rhat = {name: float(split_gelman_rubin(np.asarray(draws[name]))) for name in names}
    ess = {name: float(effective_sample_size(np.asarray(draws[name]))) for name in names}
    samples = np.stack([np.asarray(draws[name], dtype=float).reshape(-1) for name in names], axis=1)
    success = all(value < RHAT_LIMIT for value in rhat.values())
    return samples, {'rhat': rhat, 'ess': ess}, success


def _fit_mcmc(
    objective: _CalibrationObjective,
    loss: str,
    n_samples: int,
    chains: int,
    random_state: int | None,
) -> tuple[Array, dict[str, object], bool]:
    """Return the samples of MCMC with numpyro, their diagnostics, and convergence.

    A deterministic differentiable system with the Gaussian likelihood runs NUTS on its JAX run. Every other system runs
    the affine invariant ensemble sampler, with the log-likelihood evaluated on the host, and at least two walkers per
    parameter and two more. The host returns the log-likelihood in single precision, since XLA may run the callback on a
    thread of its own, where the double precision that `jax.enable_x64` sets for the calling thread does not hold.
    """
    names = objective.names
    priors = [_numpyro_prior(name, prior) for name, prior in objective.priors.items()]
    count = objective.observed.size if objective.series else 0
    tags = objective.build([float(prior.median()) for prior in objective.priors.values()]).get_tags()
    differentiable = loss == 'likelihood' and tags.differentiable and tags.randomness == 'deterministic'
    with jax.enable_x64(True):
        if differentiable:
            predict, observed = objective.get_traced(), jnp.asarray(objective.observed)

            def model() -> None:
                theta = jnp.stack([numpyro.sample(name, prior) for name, prior in zip(names, priors, strict=True)])
                residuals = predict(theta) - observed
                numpyro.factor('likelihood', -0.5 * count * jnp.log(jnp.sum(residuals**2)))

            sampler, walkers, method = NUTS(model), chains, 'sequential'
        else:
            evaluate = objective.log_likelihood if loss == 'likelihood' else objective.get_event_log_likelihood

            def host(theta: Array) -> Array:
                return np.asarray(evaluate(np.asarray(theta, dtype=float)), dtype=np.float32)

            def model() -> None:
                theta = jnp.stack([numpyro.sample(name, prior) for name, prior in zip(names, priors, strict=True)])
                shape = jax.ShapeDtypeStruct((), jnp.float32)
                numpyro.factor('likelihood', jax.pure_callback(host, shape, theta, vmap_method='sequential'))

            sampler, walkers, method = AIES(model), max(chains, 2 * len(names) + 2), 'vectorized'
        mcmc = MCMC(
            sampler,
            num_warmup=n_samples,
            num_samples=n_samples,
            num_chains=walkers,
            chain_method=method,
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(0 if random_state is None else random_state))
        draws = mcmc.get_samples(group_by_chain=True)
    samples, diagnostics, success = _chain_diagnostics(draws, names)
    diagnostics['sampler'] = 'nuts' if differentiable else 'aies'
    return samples, diagnostics, success


def _fit_variational(
    objective: _CalibrationObjective,
    loss: str,
    n_samples: int,
    budget: int,
    random_state: int | None,
) -> tuple[Array, dict[str, object], bool]:
    """Return draws of a variational posterior fitted by numpyro SVI, and its losses.

    The fit goes through the JAX run of the system.

    The guide is a multivariate normal on the unconstrained parameters, fitted by Adam. The model adds the negative
    loss, the Gaussian log-likelihood or the negative sum of squares, so the ELBO is the expected loss plus the
    divergence from the prior. A system that draws runs with a fresh JAX key at every step of the fit, so the ELBO also
    averages over the draws.
    """
    names = objective.names
    priors = [_numpyro_prior(name, prior) for name, prior in objective.priors.items()]
    count = objective.observed.size
    with jax.enable_x64(True):
        predict, observed = objective.get_traced(), jnp.asarray(objective.observed)

        def model() -> None:
            theta = jnp.stack([numpyro.sample(name, prior) for name, prior in zip(names, priors, strict=True)])
            squares = jnp.sum((predict(theta, numpyro.prng_key()) - observed) ** 2)
            score = -0.5 * count * jnp.log(squares) if loss == 'likelihood' else -squares
            numpyro.factor('loss', score)

        guide = AutoMultivariateNormal(model)
        optimizer = numpyro.optim.optax_to_numpyro(optax.adam(LEARNING_RATE))
        key, drawing = jax.random.split(jax.random.PRNGKey(0 if random_state is None else random_state))
        fit = SVI(model, guide, optimizer, Trace_ELBO()).run(key, budget, progress_bar=False)
        draws = guide.sample_posterior(drawing, fit.params, sample_shape=(n_samples,))
    losses = np.asarray(fit.losses, dtype=float)
    samples = np.stack([np.asarray(draws[name], dtype=float) for name in names], axis=1)
    return samples, {'losses': losses}, bool(np.isfinite(losses[-1]))


def _check_differentiable(objective: _CalibrationObjective) -> None:
    """Raise an error for variational calibration without the tag `differentiable`."""
    system = objective.build([float(prior.median()) for prior in objective.priors.values()])
    if not system.get_tags().differentiable:
        others = ', '.join(method for method in CalibrationMethodKind if method != CalibrationMethodKind.VARIATIONAL)
        error_msg = (
            f'Variational calibration differentiates runs, but {type(system).__name__} does not hold the tag '
            f'differentiable. Use one of the methods {others}, which do not need it.'
        )
        raise ValidationError(error_msg)


def _posterior_result(
    objective: _CalibrationObjective,
    loss: str,
    samples: Array,
    weights: Array,
    *,
    diagnostics: dict[str, object],
    success: bool,
    random_state: int | None,
) -> CalibrationResult:
    """Return the result of a posterior method with its medians and covariance.

    It also holds the predictive check.
    """
    estimate = np.asarray([_weighted_median(column, weights) for column in samples.T])
    covariance = np.atleast_2d(np.cov(samples.T, aweights=weights)) if len(samples) > 1 else None
    predictive = objective.get_predictive(samples, weights, loss, np.random.default_rng(random_state))
    return _result(
        objective,
        estimate,
        covariance,
        success,
        samples={name: samples[:, index] for index, name in enumerate(objective.names)},
        weights=weights,
        diagnostics=diagnostics,
        predictive=predictive,
    )


def _result(
    objective: _CalibrationObjective,
    estimate: Array,
    covariance: Array | None,
    success: bool,
    *,
    samples: dict[str, Array] | None = None,
    weights: Array | None = None,
    diagnostics: dict[str, object] | None = None,
    predictive: dict[str, object] | None = None,
) -> CalibrationResult:
    """Return the result of a calibration at an estimate, with its goodness of fit."""
    simulated = objective.get_replayed(estimate) if objective.recorded is not None else objective.simulate(estimate)
    final = np.concatenate([simulated[path] - observed for path, (_, observed) in objective.series.items()] or [[]])
    observed = objective.observed if objective.series else np.zeros(0)
    nonzero = observed != 0
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = float(1 - np.sum(final**2) / np.sum((observed - observed.mean()) ** 2)) if observed.size else np.nan
        rmse = float(np.sqrt(np.mean(final**2))) if final.size else np.nan
        mae = float(np.mean(np.abs(final))) if final.size else np.nan
    return CalibrationResult(
        system=objective.build(estimate),
        estimates=dict(zip(objective.names, map(float, estimate), strict=True)),
        r2=r2,
        rmse=rmse,
        mae=mae,
        mape=float(100 * np.mean(np.abs(final[nonzero] / observed[nonzero]))) if nonzero.any() else float('nan'),
        covariance=covariance,
        simulated=simulated,
        success=success,
        samples=samples,
        weights=weights,
        diagnostics=diagnostics or {},
        predictive=predictive,
    )


def calibrate(
    system: Source,
    data: Mapping[str, tuple[npt.ArrayLike, npt.ArrayLike]] | Trajectory,
    bounds: Bounds,
    *,
    method: str = 'local',
    initial: Mapping[str, float] | None = None,
    loss: str = 'squares',
    summary: Summary | None = None,
    n_samples: int = 1000,
    budget: int | None = None,
    chains: int = 4,
    replicates: int = 1,
    dt: float | None = None,
    solver: BaseSolver | None = None,
    random_state: int | None = None,
) -> CalibrationResult:
    """Return the fit of the model parameters of a system to observed time series.

    The fitted system reproduces the observed time series. It is a copy of the system with `set_params` applied, or
    what the factory builds from the parameters. Every evaluation runs the system through `simulate` to the last data
    time and interpolates its states or observables at the data times. Stochastic systems average `replicates` runs
    with fixed streams, so the loss is smooth in the parameters.

    `method` says how to search or sample and `loss` what scores a candidate:

    - `'local'`: a local SciPy fit, `least_squares` for `'squares'` and `minimize` for `'likelihood'` and `'events'`,
      with the covariance from the Jacobian of the residuals.
    - `'global'`: the best of `budget` evaluations, 200 by default, chosen by Optuna's seeded tree-structured Parzen
      estimator over the bounds, which copes with the rough losses of stochastic runs.
    - `'abc_smc'`: an approximate posterior by sequential Monte Carlo approximate Bayesian computation with pyabc,
      which compares the summaries of every simulation with the observed ones by an adaptive distance and stops after
      `budget` simulations, 20000 by default. Its diagnostics hold the tolerance and the acceptance rate of every
      generation.
    - `'mcmc'`: `chains` chains of `n_samples` draws of the posterior of the Gaussian or the event likelihood with
      numpyro, by NUTS through the JAX run of a system with the tag `differentiable` and the Gaussian likelihood, and
      by the gradient-free affine invariant ensemble sampler otherwise. Its diagnostics hold the split R-hat and the
      effective sample size of every parameter.
    - `'variational'`: a multivariate normal posterior fitted by numpyro SVI over `budget` steps, 2000 by default,
      through the JAX run of a system with the tag `differentiable`. With `'squares'` it is the generalised posterior
      of the sum of squares. Its diagnostics hold the loss history.

    A posterior method returns the samples, their weights, the posterior medians as the estimates, and a posterior
    predictive check of at most 200 draws at the data times.

    A bound is a pair `(low, high)`, read as a uniform prior, or a SciPy distribution, read as the prior of that
    parameter. Point fits search the pair or the central 99.9% of the distribution. `'squares'` is least squares,
    `'likelihood'` the Gaussian likelihood with an unknown variance, and `'events'` the log-probability of the draws of
    a trajectory recorded with `simulate(..., record_draws=True)`, given as `data`, replayed under every candidate with
    `random_state`, the seed of the recorded run.

    It appears in many fields. It fits the rates of an epidemic to case counts, the rate constants of a reaction to
    measured concentrations, and the parameters of a predator and prey model to pelt records.

    Args:
        system:
            The system, whose copies take the model parameters through `set_params`, or a factory of systems, such as
            a system class, which takes them as keyword arguments.

        data:
            A mapping from state or observable path to its observed `(times, values)`, or a trajectory recorded with
            its draws for the loss `'events'`.

        bounds:
            A mapping from model parameter of the system to its `(low, high)` bounds or its SciPy prior.

        method:
            `'local'`, `'global'`, `'abc_smc'`, `'mcmc'`, or `'variational'`.

        initial:
            The starting values of the parameters of a local fit, or `None` for the medians of their priors.

        loss:
            `'squares'`, `'likelihood'`, or `'events'`, as the method allows.

        summary:
            The function from simulated or observed series by path to summary statistics that ABC-SMC compares, or
            `None` for the concatenated series.

        n_samples:
            The number of samples of a posterior method, per chain for MCMC.

        budget:
            The number of evaluations of a global fit, simulations of ABC-SMC, or steps of variational calibration, or
            `None` for the default of the method.

        chains:
            The number of chains of MCMC.

        replicates:
            The number of runs averaged per evaluation.

        dt:
            The time between frames, or `None` for the tuned one.

        solver:
            The solver, or `None` for the tuned one.

        random_state:
            The seed of the runs and of the method.

    Returns:
        result:
            The fitted system, the estimates, and the goodness of fit.

    Raises:
        ValidationError:
            The system is a system name, the method or the loss is unknown or they do not fit, there is no data, the
            replicates are bad, or a bound is neither a pair nor a SciPy distribution or has a dotted path.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.experiments import calibrate
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> times = np.arange(0, 21.0)
        >>> values = 100 / (1 + 9 * np.exp(-0.5 * times))
        >>> bounds = {'r': (0.1, 2.0), 'K': (50, 200)}
        >>> fitted = calibrate(LogisticGrowthSystem(), {'population.x': (times, values)}, bounds).system
        >>> round(fitted.r, 2), round(fitted.K)
        (0.5, 100)
        >>> result = calibrate(LogisticGrowthSystem(), {'population.x': (times, values)}, bounds, method='global',
        ...     budget=100, random_state=0)
        >>> bool(abs(result.estimates['r'] - 0.5) < 0.1)
        True
        >>> noisy = {'population.x': (times, values * (1 + 0.02 * np.random.default_rng(0).standard_normal(21)))}
        >>> posterior = calibrate(LogisticGrowthSystem(), noisy, bounds, method='variational', loss='likelihood',
        ...     n_samples=200, random_state=0)
        >>> round(posterior.estimates['r'], 1), posterior.samples['r'].shape
        (0.5, (200,))
    """
    _check_source(system)
    _check_settings(method, loss, data, bounds, replicates)
    priors = _check_priors(bounds)
    objective = _CalibrationObjective(
        system,
        _series_of(data),
        priors,
        replicates=replicates,
        dt=dt,
        solver=solver,
        random_state=random_state,
        recorded=data if loss == 'events' and isinstance(data, Trajectory) else None,
        summary=summary,
    )
    budget = BUDGETS[method] if budget is None else budget
    _check_count('n_samples', n_samples)
    _check_count('chains', chains)
    if method == 'variational':
        _check_differentiable(objective)
    if method in {'mcmc', 'variational'}:
        for name, prior in priors.items():
            _numpyro_prior(name, prior)
    if method == 'abc_smc':
        samples, weights, diagnostics = _fit_abc(objective, n_samples, budget, random_state)
        return _posterior_result(
            objective,
            loss,
            samples,
            weights,
            diagnostics=diagnostics,
            success=True,
            random_state=random_state,
        )
    if method in {'mcmc', 'variational'}:
        if method == 'mcmc':
            samples, diagnostics, success = _fit_mcmc(objective, loss, n_samples, chains, random_state)
        else:
            samples, diagnostics, success = _fit_variational(objective, loss, n_samples, budget, random_state)
        weights = np.full(len(samples), 1 / len(samples))
        return _posterior_result(
            objective,
            loss,
            samples,
            weights,
            diagnostics=diagnostics,
            success=success,
            random_state=random_state,
        )
    if method == 'global':
        estimate = _fit_global(objective, loss, budget, random_state)
        return _result(objective, estimate, objective.get_covariance(estimate, loss), success=True)
    start = np.asarray(
        [(initial or {}).get(name, float(prior.median())) for name, prior in priors.items()],
        dtype=float,
    )
    estimate, success = _fit_local(objective, loss, start)
    return _result(objective, estimate, objective.get_covariance(estimate, loss), success)
