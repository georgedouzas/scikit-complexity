"""Check the credible intervals of calibration on data simulated from prior draws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections.abc import Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..engine.runtime import simulate
from ..engine.solvers import BaseSolver
from ..errors import ValidationError
from ..types import Array
from ._calibration import LOSSES, CalibrationResult, Summary, _check_priors, _series_of, calibrate
from ._runs import EXECUTORS, _check_count, _check_source
from ._utils import build_system
from .types import Bounds, Source

POSTERIOR_METHODS = ('abc_smc', 'mcmc', 'variational')


@dataclass(frozen=True)
class CoverageResult:
    """Empirical coverage of the credible intervals of calibration on data simulated from prior draws.

    A calibrated posterior covers the true value with its central interval of level `q` in a share `q` of the data
    sets, and the ranks of the true values among their samples are uniform, as simulation-based calibration checks.

    Args:
        levels:
            The levels of the central credible intervals.

        coverage:
            The share of the data sets whose interval holds the true value, by level and parameter.

        ranks:
            The weighted share of the samples below the true value, one per data set, by parameter.

        truths:
            The true value of every data set, by parameter.

        results:
            The calibration of every data set.

    Examples:
        >>> from skcomplex.core.experiments import CoverageResult
        >>> CoverageResult((0.9,), {0.9: {'r': 0.85}}, {}, {}, ()).coverage[0.9]['r']
        0.85
    """

    levels: tuple[float, ...]
    coverage: dict[float, dict[str, float]]
    ranks: dict[str, Array]
    truths: dict[str, Array]
    results: tuple[CalibrationResult, ...]


def _interval(values: Array, weights: Array, level: float) -> tuple[float, float]:
    """Return the central interval of a level of weighted samples."""
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / np.sum(weights)
    tail = (1 - level) / 2
    ordered = values[order]
    low = ordered[min(np.searchsorted(cumulative, tail), ordered.size - 1)]
    high = ordered[min(np.searchsorted(cumulative, 1 - tail), ordered.size - 1)]
    return float(low), float(high)


def _calibrate_truth(
    truth: dict[str, float],
    seed: int,
    *,
    source: Source,
    bounds: Bounds,
    times: Array,
    paths: Sequence[str] | None,
    noise: float,
    method: str,
    loss: str,
    summary: Summary | None,
    n_samples: int,
    budget: int | None,
    chains: int,
    dt: float | None,
    solver: BaseSolver | None,
) -> CalibrationResult:
    """Return the calibration of data simulated at one true parameter vector, with Gaussian noise."""
    rng = np.random.default_rng(seed)
    trajectory = simulate(build_system(source, truth), until=float(times.max()), dt=dt, solver=solver, random_state=rng)
    names = list(paths) if paths is not None else list(_series_of(trajectory))
    data = {}
    for path in names:
        values = np.interp(times, trajectory.times, np.asarray(trajectory.get_values(path), dtype=float))
        data[path] = (times, values + noise * rng.standard_normal(times.size))
    return calibrate(
        source,
        data,
        bounds,
        method=method,
        loss=loss,
        summary=summary,
        n_samples=n_samples,
        budget=budget,
        chains=chains,
        dt=dt,
        solver=solver,
        random_state=int(rng.integers(2**31)),
    )


def posterior_coverage(
    system: Source,
    bounds: Bounds,
    times: npt.ArrayLike,
    *,
    paths: Sequence[str] | None = None,
    method: str = 'mcmc',
    loss: str = 'likelihood',
    n: int = 40,
    levels: Sequence[float] = (0.5, 0.9),
    noise: float = 0.0,
    summary: Summary | None = None,
    n_samples: int = 500,
    budget: int | None = None,
    chains: int = 4,
    dt: float | None = None,
    solver: BaseSolver | None = None,
    random_state: int | None = None,
    executor: str | Executor = 'serial',
) -> CoverageResult:
    """Return the empirical coverage of the credible intervals of a posterior method over prior draws.

    It draws `n` true parameter vectors from the priors the bounds give, simulates data at each at the given times with
    Gaussian noise of a standard deviation, calibrates every data set with the posterior method, and counts how often
    the central credible interval of every level holds the true value. It also gives the rank of every true value among
    its samples, the check of simulation-based calibration (Talts et al. 2018).

    Args:
        system:
            The system, or a factory of systems.

        bounds:
            A pair `(low, high)` or a SciPy prior per model parameter, as `calibrate` takes them.

        times:
            The times of the simulated data.

        paths:
            The state or observable paths of the data, or `None` for every scalar state.

        method:
            `'abc_smc'`, `'mcmc'`, or `'variational'`.

        loss:
            The loss of the method.

        n:
            The number of true parameter vectors.

        levels:
            The levels of the central credible intervals, between 0 and 1.

        noise:
            The standard deviation of the Gaussian noise added to the simulated data.

        summary:
            The summary statistics of ABC-SMC, or `None`.

        n_samples:
            The number of samples of every calibration, per chain for MCMC.

        budget:
            The budget of every calibration, or `None` for the default of the method.

        chains:
            The number of chains of MCMC.

        dt:
            The time between frames, or `None` for the tuned one.

        solver:
            The solver, or `None` for the tuned one.

        random_state:
            The seed of the true values, the noise, and the calibrations.

        executor:
            `'serial'`, `'thread'`, `'process'`, or a `concurrent.futures` executor that runs the calibrations. MCMC
            and variational calibration run numpyro, whose effect handlers are global, so they do not run on threads.

    Returns:
        result:
            The coverage by level and parameter, the ranks, the true values, and the calibrations.

    Raises:
        ValidationError:
            The method is not a posterior method, the loss does not fit it, a level is not between 0 and 1, `n` is not
            a positive integer, or the executor is unknown or runs numpyro on threads.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.experiments import posterior_coverage
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> result = posterior_coverage(
        ...     LogisticGrowthSystem(n=10.0),
        ...     {'r': (0.5, 1.0)},
        ...     np.arange(0.0, 11.0),
        ...     method='variational',
        ...     n=3,
        ...     levels=(0.9,),
        ...     noise=5.0,
        ...     n_samples=100,
        ...     budget=300,
        ...     random_state=0,
        ... )
        >>> sorted(result.coverage[0.9]), result.ranks['r'].shape
        (['r'], (3,))
    """
    source = _check_source(system)
    if method not in POSTERIOR_METHODS:
        error_msg = f'Posterior coverage calibrates by a posterior method. Use one of {", ".join(POSTERIOR_METHODS)}.'
        raise ValidationError(error_msg)
    if loss not in LOSSES[method] or loss == 'events':
        allowed = ', '.join(item for item in LOSSES[method] if item != 'events')
        error_msg = f'Posterior coverage with the method {method!r} takes the losses {allowed}.'
        raise ValidationError(error_msg)
    _check_count('n', n)
    checked = tuple(float(level) for level in levels)
    if not checked or any(not 0 < level < 1 for level in checked):
        error_msg = f'The levels of posterior coverage must lie between 0 and 1, not {tuple(levels)!r}.'
        raise ValidationError(error_msg)
    if isinstance(executor, str) and executor not in EXECUTORS:
        error_msg = f"executor = {executor!r} must be 'serial', 'thread', 'process', or an executor."
        raise ValidationError(error_msg)
    if (executor == 'thread' or isinstance(executor, ThreadPoolExecutor)) and method != 'abc_smc':
        error_msg = (
            f'The method {method!r} runs numpyro, whose effect handlers are global, so it cannot run on threads. Use '
            "executor='process' or executor='serial'."
        )
        raise ValidationError(error_msg)
    priors = _check_priors(bounds)
    rng = np.random.default_rng(random_state)
    draws = {name: np.asarray(prior.rvs(size=n, random_state=rng), dtype=float) for name, prior in priors.items()}
    truths = [{name: float(values[index]) for name, values in draws.items()} for index in range(n)]
    seeds = [int(seed) for seed in rng.integers(2**31, size=n)]
    task = functools.partial(
        _calibrate_truth,
        source=source,
        bounds=bounds,
        times=np.asarray(times, dtype=float),
        paths=paths,
        noise=noise,
        method=method,
        loss=loss,
        summary=summary,
        n_samples=n_samples,
        budget=budget,
        chains=chains,
        dt=dt,
        solver=solver,
    )
    if executor == 'serial':
        results = tuple(task(truth, seed) for truth, seed in zip(truths, seeds, strict=True))
    else:
        pool = executor if isinstance(executor, Executor) else None
        pool = pool or (ThreadPoolExecutor() if executor == 'thread' else ProcessPoolExecutor())
        try:
            results = tuple(pool.map(task, truths, seeds))
        finally:
            if isinstance(executor, str):
                pool.shutdown()
    coverage: dict[float, dict[str, float]] = {level: {} for level in checked}
    ranks: dict[str, Array] = {}
    for name in priors:
        pairs = [((result.samples or {})[name], np.asarray(result.weights)) for result in results]
        values = [truth[name] for truth in truths]
        below = [np.sum(weights[samples < value]) for (samples, weights), value in zip(pairs, values, strict=True)]
        ranks[name] = np.asarray(below) / np.asarray([np.sum(weights) for _, weights in pairs])
        for level in checked:
            intervals = [_interval(samples, weights, level) for samples, weights in pairs]
            inside = [low <= value <= high for (low, high), value in zip(intervals, values, strict=True)]
            coverage[level][name] = float(np.mean(inside))
    return CoverageResult(checked, coverage, ranks, draws, results)
