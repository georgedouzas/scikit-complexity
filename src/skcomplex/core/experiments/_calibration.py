"""Fit the model parameters of a system to observed time series."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import optuna
from scipy import stats
from scipy.optimize import least_squares, minimize

from ..components.system import BaseSystem
from ..engine.runtime import Trajectory, simulate
from ..engine.solvers import BaseSolver
from ..errors import ValidationError
from ..types import Array
from ._designs import _check_bounds
from ._runs import _build, _check_count, _check_source
from .types import Bounds, Source

METHODS = ('local', 'global', 'abc_smc', 'mcmc', 'variational')
LOSSES = {
    'local': ('squares', 'likelihood', 'events'),
    'global': ('squares', 'likelihood', 'events'),
    'abc_smc': ('squares',),
    'mcmc': ('likelihood', 'events'),
    'variational': ('likelihood', 'squares'),
}
BUDGETS = {'local': 0, 'global': 200, 'abc_smc': 20000, 'mcmc': 0, 'variational': 2000}
CENTRAL = (0.0005, 0.9995)

type Series = dict[str, tuple[Array, Array]]
type Summary = Callable[[Mapping[str, Array]], npt.ArrayLike]


@dataclass(frozen=True)
class CalibrationResult:
    """Fit of the model parameters of a system to observed time series.

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
        return _build(self.source, dict(zip(self.names, map(float, values), strict=True)))

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
            runs.append(self._at_times(trajectory))
        return {path: np.mean([run[path] for run in runs], axis=0) for path in self.series}

    def _at_times(self, trajectory: Trajectory) -> dict[str, Array]:
        """Return the values of a trajectory interpolated at the data times by path."""
        return {
            path: np.interp(times, trajectory.times, np.asarray(trajectory.values(path), dtype=float))
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

    def replayed(self, values: Sequence[float]) -> dict[str, Array]:
        """Return the replayed values at the data times by path."""
        return self._at_times(self._replay(values))

    def residuals(self, values: Sequence[float]) -> Array:
        """Return the simulated minus the observed values of every path, concatenated."""
        simulated = self.simulate(values)
        return np.concatenate([simulated[path] - observed for path, (_, observed) in self.series.items()])

    def log_likelihood(self, values: Sequence[float]) -> float:
        """Return the Gaussian log-likelihood with the variance profiled out, up to a constant."""
        residuals = self.residuals(values)
        return float(-0.5 * residuals.size * np.log(np.sum(residuals**2)))

    def event_log_likelihood(self, values: Sequence[float]) -> float:
        """Return the log-probability of the recorded draws replayed under a parameter vector.

        A replay whose draws do not match the recording, such as a group that grows otherwise, has probability 0.
        """
        try:
            probability = self._replay(values).log_probability
        except ValidationError:
            return -np.inf
        return -np.inf if probability is None else float(probability)

    def log_prior(self, values: Sequence[float]) -> float:
        """Return the log density of the priors at a parameter vector."""
        return float(sum(prior.logpdf(value) for prior, value in zip(self.priors.values(), values, strict=True)))

    def loss(self, values: Sequence[float], loss: str) -> float:
        """Return the loss of a parameter vector to minimise.

        It is half the sum of squares, the negative Gaussian log-likelihood, or the negative event log-likelihood.
        """
        if loss == 'squares':
            return float(0.5 * np.sum(self.residuals(values) ** 2))
        if loss == 'likelihood':
            return -self.log_likelihood(values)
        return -self.event_log_likelihood(values)

    def summaries(self, simulated: Mapping[str, Array]) -> Array:
        """Return the summary statistics of series by path."""
        if self.summary is None:
            return np.concatenate([np.asarray(simulated[path], dtype=float) for path in self.series])
        return np.atleast_1d(np.asarray(self.summary(simulated), dtype=float))

    def covariance(self, estimate: Array, loss: str) -> Array | None:
        """Return the covariance of point estimates from the Jacobian of the residuals.

        It is `None` for the event likelihood, which has no residuals, and when the Jacobian is singular.
        """
        if loss == 'events':
            return None
        low, high = self.box
        fit = least_squares(self.residuals, np.clip(estimate, low, high), bounds=(low, high), max_nfev=1)
        jacobian, final = fit.jac, self.residuals(estimate)
        dof = max(final.size - len(self.names), 1)
        try:
            return np.asarray(np.linalg.inv(jacobian.T @ jacobian) * float(np.sum(final**2)) / dof)
        except np.linalg.LinAlgError:
            return None


def _check_priors(bounds: Bounds) -> dict[str, stats.rv_continuous]:
    """Return the prior of every parameter from its bounds.

    A pair `(low, high)` is the uniform prior on it, and a SciPy distribution is the prior itself.

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
    """Raise an error for a method, a loss, data, replicates, or bounds that calibration cannot take."""
    if method not in METHODS:
        error_msg = f'Unknown method {method!r}. Use one of {", ".join(METHODS)}.'
        raise ValidationError(error_msg)
    if loss not in LOSSES[method]:
        known = sorted({item for items in LOSSES.values() for item in items})
        start = f'Unknown loss {loss!r}.' if loss not in known else f'The method {method!r} does not take {loss!r}.'
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
            name: (np.asarray(data.times, dtype=float), np.asarray(data.values(name), dtype=float))
            for name in states.names
            if states.entry(name).size == 1
        }
    return {
        path: (np.asarray(times, dtype=float), np.asarray(values, dtype=float))
        for path, (times, values) in data.items()
    }


def _fit_local(objective: _CalibrationObjective, loss: str, start: Array) -> tuple[Array, bool]:
    """Return the local SciPy fit of a loss from a starting vector, and whether it converged."""
    low, high = objective.box
    if loss == 'squares':
        fit = least_squares(objective.residuals, start, bounds=(low, high), x_scale='jac')
        return np.asarray(fit.x, dtype=float), bool(fit.success)
    if loss == 'likelihood':
        count = objective.observed.size
        fit = minimize(
            lambda values: 0.5 * count * np.log(np.sum(objective.residuals(values) ** 2)),
            start,
            bounds=list(zip(low, high, strict=True)),
        )
    else:
        fit = minimize(
            lambda values: min(objective.loss(values, loss), np.finfo(float).max),
            start,
            bounds=list(zip(low, high, strict=True)),
            method='Nelder-Mead',
        )
    return np.asarray(fit.x, dtype=float), bool(fit.success)


def _fit_global(objective: _CalibrationObjective, loss: str, budget: int, random_state: int | None) -> Array:
    """Return the best of `budget` evaluations of a loss chosen by a seeded tree-structured Parzen estimator."""
    low, high = objective.box
    names = objective.names

    def evaluate(trial: optuna.Trial) -> float:
        values = [trial.suggest_float(name, lower, upper) for name, lower, upper in zip(names, low, high, strict=True)]
        return min(objective.loss(values, loss), np.finfo(float).max)

    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(logging.WARNING)
    try:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(evaluate, n_trials=budget)
    finally:
        optuna.logging.set_verbosity(verbosity)
    return np.asarray([study.best_params[name] for name in names], dtype=float)


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
    simulated = objective.replayed(estimate) if objective.recorded is not None else objective.simulate(estimate)
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

    A bound is a pair `(low, high)`, read as a uniform prior, or a SciPy distribution, read as the prior of that
    parameter. Point fits search the pair or the central 99.9% of the distribution. `'squares'` is least squares,
    `'likelihood'` the Gaussian likelihood with an unknown variance, and `'events'` the log-probability of the draws of
    a trajectory recorded with `simulate(..., record_draws=True)`, given as `data`, replayed under every candidate with
    `random_state`, the seed of the recorded run.

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
        >>> fitted = calibrate(LogisticGrowthSystem(), {'population.n': (times, values)}, bounds).system
        >>> round(fitted.r, 2), round(fitted.K)
        (0.5, 100)
        >>> result = calibrate(LogisticGrowthSystem(), {'population.n': (times, values)}, bounds, method='global',
        ...     budget=100, random_state=0)
        >>> bool(abs(result.estimates['r'] - 0.5) < 0.1)
        True
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
    if method == 'global':
        estimate = _fit_global(objective, loss, budget, random_state)
        return _result(objective, estimate, objective.covariance(estimate, loss), success=True)
    start = np.asarray(
        [(initial or {}).get(name, float(prior.median())) for name, prior in priors.items()],
        dtype=float,
    )
    estimate, success = _fit_local(objective, loss, start)
    return _result(objective, estimate, objective.covariance(estimate, loss), success)
