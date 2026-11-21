"""Tell which model parameters of a system its data determine."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import logging
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pypesto
import pypesto.optimize
import pypesto.profile

from ..engine.solvers import BaseSolver
from ..errors import ValidationError
from ..types import Array
from ._calibration import SEED_LIMIT, _CalibrationObjective, _check_priors, _check_settings, _series_of, calibrate
from ._runs import _check_count, _check_source, _map_calls
from .types import Bounds, SystemSource

METHODS = ('profile', 'refits')
WEAK_FACTOR = 10.0
VERDICTS = ('strong', 'weak', 'none')
OPTIMIZER = 'L-BFGS-B'
FD_STEP = 1e-5


type Data = Mapping[str, tuple[npt.ArrayLike, npt.ArrayLike]]


@dataclass(frozen=True)
class IdentifiabilityResult:
    """A result of `identifiability`, the best fit and a verdict per parameter.

    It holds the best fit and whether the data pin down each parameter, such as `{'r': 'strong'}`. A profile also holds
    the profile and the interval of every parameter, with the verdicts `'identifiable'` and `'not identifiable'`. Refits
    hold the average error of every parameter per level of noise, with the verdicts `'strong'`, `'weak'`, and `'none'`.
    It tells whether case counts fix the rates of an epidemic, counts fix the capacity of a population, and blood
    samples fix the clearance of a drug.

    Read more in the user guide, under Calibrate.

    Args:
        estimates:
            The best-fit value of every parameter.

        verdicts:
            The verdict of every parameter.

        profiles:
            The values of the profiled parameter and the likelihood ratio along its profile, by parameter, or `None`
            for refits.

        intervals:
            The `(low, high)` likelihood-ratio interval of every parameter, with `nan` for a side that reaches its
            bound, or `None` for refits.

        errors:
            The average relative error of every parameter by noise level, or `None` for a profile.

    See Also:
        identifiability: Returns this result.

    Examples:
        >>> from skcomplex.core.experiments import IdentifiabilityResult
        >>> IdentifiabilityResult({'r': 0.5}, {'r': 'strong'}, None, None, {'r': {0.01: 0.004}}).verdicts
        {'r': 'strong'}
    """

    estimates: dict[str, float]
    verdicts: dict[str, str]
    profiles: dict[str, tuple[Array, Array]] | None
    intervals: dict[str, tuple[float, float]] | None
    errors: dict[str, dict[float, float]] | None


def _get_interval(values: Array, ratios: Array, threshold: float) -> tuple[float, float]:
    """Return the interval where the likelihood ratio of a profile stays high.

    The ratio stays at or above the threshold inside it. Each end is interpolated linearly between the two profile
    points around the crossing, and an end whose side of the profile never falls below the threshold is `nan`.
    """
    order = np.argsort(values)
    values, ratios = values[order], ratios[order]
    best = int(np.argmax(ratios))
    ends = []
    for side in (np.arange(best, -1, -1), np.arange(best, values.size)):
        below = np.flatnonzero(ratios[side] < threshold)
        if below.size == 0:
            ends.append(float('nan'))
            continue
        outer, inner = side[below[0]], side[below[0] - 1]
        share = (ratios[inner] - threshold) / (ratios[inner] - ratios[outer])
        ends.append(float(values[inner] + share * (values[outer] - values[inner])))
    return ends[0], ends[1]


def _get_profiles(
    objective: _CalibrationObjective,
    level: float,
    n_starts: int,
    random_state: int | None,
) -> IdentifiabilityResult:
    """Return the profile likelihood of every parameter with pyPESTO.

    Every parameter also gets its interval and its verdict. The best fit is the best of `n_starts` local fits of the
    negative Gaussian log-likelihood with finite-difference gradients. pyPESTO works on the bounds scaled to the unit
    box, so its steps and differences are shares of the range of every parameter. pyPESTO draws its start points from
    the global NumPy random state, which is seeded from `random_state` and restored after, and its log is silenced to
    warnings.
    """
    low, high = objective.box
    width = high - low
    function = pypesto.Objective(fun=lambda unit: -objective.log_likelihood(low + unit * width))
    gradient = pypesto.objective.FD(function, delta_fun=FD_STEP)
    problem = pypesto.Problem(gradient, np.zeros(low.size), np.ones(low.size), x_names=objective.names)
    optimizer = pypesto.optimize.ScipyOptimizer(method=OPTIMIZER)
    threshold = float(pypesto.profile.chi2_quantile_to_ratio(level))
    options = pypesto.profile.ProfileOptions(ratio_min=threshold / 2)
    state = np.random.get_state()
    logger = logging.getLogger('pypesto')
    verbosity = logger.level
    np.random.seed(np.random.default_rng(random_state).integers(SEED_LIMIT))
    logger.setLevel(logging.ERROR)
    try:
        result = pypesto.optimize.minimize(problem, optimizer, n_starts=n_starts, progress_bar=False)
        result = pypesto.profile.parameter_profile(
            problem,
            result,
            optimizer,
            profile_options=options,
            progress_bar=False,
        )
    finally:
        np.random.set_state(state)
        logger.setLevel(verbosity)
    best = low + np.asarray(result.optimize_result[0].x, dtype=float) * width
    profiles, intervals, verdicts = {}, {}, {}
    for index, (name, profile) in enumerate(zip(objective.names, result.profile_result.list[0], strict=True)):
        values = low[index] + np.asarray(profile.x_path[index], dtype=float) * width[index]
        ratios = np.asarray(profile.ratio_path, dtype=float)
        profiles[name] = (values, ratios)
        intervals[name] = _get_interval(values, ratios, threshold)
        verdicts[name] = 'not identifiable' if np.isnan(intervals[name]).any() else 'identifiable'
    estimates = dict(zip(objective.names, map(float, best), strict=True))
    return IdentifiabilityResult(estimates, verdicts, profiles, intervals, None)


def _get_verdict(error: float, level: float) -> str:
    """Return the verdict of an average relative error at a noise level.

    The thresholds are those of Liyanage et al. (2025).
    """
    if error <= level:
        return VERDICTS[0]
    if error <= WEAK_FACTOR * level:
        return VERDICTS[1]
    return VERDICTS[2]


def _get_refits(
    source: SystemSource,
    data: Data,
    bounds: Bounds,
    *,
    noise: Sequence[float],
    n: int,
    dt: float | None,
    solver: BaseSolver | None,
    random_state: int | None,
    executor: str | Executor,
) -> IdentifiabilityResult:
    """Return the average relative errors of refits to noisy copies of the best fit.

    It also returns the verdicts. Every copy adds Gaussian noise of a relative level to the values the best fit
    simulates at the data times, and is refitted from the best fit. A parameter takes its worst verdict over the levels.
    """
    best = calibrate(source, data, bounds, dt=dt, solver=solver, random_state=random_state)
    series = _series_of(data)
    rng = np.random.default_rng(random_state)
    fit = functools.partial(
        calibrate,
        source,
        bounds=bounds,
        initial=best.estimates,
        dt=dt,
        solver=solver,
        random_state=random_state,
    )
    errors: dict[str, dict[float, float]] = {name: {} for name in best.estimates}
    verdicts = dict.fromkeys(best.estimates, VERDICTS[0])
    for level in noise:
        copies = [
            {
                path: (times, best.simulated[path] * (1 + level * rng.standard_normal(times.size)))
                for path, (times, _) in series.items()
            }
            for _ in range(n)
        ]
        fits = _map_calls(fit, copies, executor)
        for name, value in best.estimates.items():
            relative = [abs(result.estimates[name] - value) / abs(value) for result in fits]
            errors[name][level] = float(np.mean(relative))
            verdict = _get_verdict(errors[name][level], level)
            verdicts[name] = max(verdicts[name], verdict, key=VERDICTS.index)
    return IdentifiabilityResult(best.estimates, verdicts, None, None, errors)


def identifiability(
    system: SystemSource,
    data: Data,
    bounds: Bounds,
    *,
    method: str = 'profile',
    level: float = 0.95,
    n_starts: int = 5,
    noise: Sequence[float] = (0.01,),
    n: int = 100,
    dt: float | None = None,
    solver: BaseSolver | None = None,
    random_state: int | None = None,
    executor: str | Executor = 'serial',
) -> IdentifiabilityResult:
    """Return which model parameters of a system the observed time series determine.

    A calibration returns estimates even when the data cannot pin them down, and this function tells whether they do. A
    profile fixes each parameter at values around the best fit and fits the others again, and the parameter is
    identifiable when the fit worsens enough on both sides within its bounds. Refits add noise to the best run, fit
    again, and judge each parameter by how far its estimates stray. It tells whether counts fix the growth rate and
    capacity of a population, whether case counts fix the rates of an epidemic, and whether blood samples fix the
    clearance of a drug.

    Read more in the user guide, under Calibrate.

    Args:
        system:
            The system, whose clones take the parameters through `set_params`, or a factory of systems.

        data:
            A mapping from state or observable path to its observed `(times, values)`.

        bounds:
            A mapping from model parameter to its `(low, high)` bounds or its SciPy prior, as `calibrate` takes them.
            The profiles stop at the bounds.

        method:
            `'profile'` or `'refits'`. A profile, with pyPESTO, takes the best of `n_starts` local fits of the
            Gaussian likelihood, then fixes each parameter on both sides of it and fits the others again until the
            likelihood ratio falls below the threshold of `level` or the bound is reached. A parameter is
            `'identifiable'` when both ends of its interval lie inside its bounds. Refits, after Liyanage et al.
            (2025), add noise at every level of `noise` to `n` copies of the best run and fit each again. A parameter
            is `'strong'` when its average relative error is at most the level, `'weak'` when it is at most ten times
            the level, and `'none'` above, taking its worst verdict over the levels.

        level:
            The confidence level of the likelihood-ratio intervals of a profile, between 0 and 1.

        n_starts:
            The number of local fits of a profile whose best is the best fit.

        noise:
            The relative noise levels of the refits, such as 0.01 for 1% of every value.

        n:
            The number of noisy copies per level of the refits.

        dt:
            The time between frames, or `None` for the tuned one.

        solver:
            The solver, or `None` for the tuned one.

        random_state:
            The seed of the start points, the runs, and the noise.

        executor:
            `'serial'`, `'thread'`, `'process'`, or a `concurrent.futures` executor that runs the refits.

    Returns:
        result:
            The best fit, the verdicts, and the profiles and intervals or the relative errors.

    Raises:
        ValidationError:
            The method is unknown, the level is not between 0 and 1, a noise level is not positive, a count is not a
            positive integer, there is no data, or a bound is not valid.

        ExperimentError:
            The system is a name.

    See Also:
        calibrate: Fits the parameters of a system to data.

    Examples:
        >>> from skcomplex.core.experiments import identifiability
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> trajectory = LogisticGrowthSystem(r=0.5).prepare().simulate(until=20, dt=1.0)
        >>> data = {'population.x': (trajectory.times, trajectory.get_values('population.x'))}
        >>> identifiability(LogisticGrowthSystem(), data, {'r': (0.1, 2.0), 'K': (50.0, 500.0)}).verdicts
        {'r': 'identifiable', 'K': 'identifiable'}
    """
    source = _check_source(system)
    if method not in METHODS:
        error_msg = f'Unknown method {method!r}. Use one of {", ".join(repr(item) for item in METHODS)}.'
        raise ValidationError(error_msg)
    if not 0 < level < 1:
        error_msg = f'The level of the intervals must lie between 0 and 1, not {level!r}.'
        raise ValidationError(error_msg)
    if not noise or any(not float(item) > 0 for item in noise):
        error_msg = f'The noise levels of the refits must be positive, not {tuple(noise)!r}.'
        raise ValidationError(error_msg)
    _check_count('n', n)
    _check_count('n_starts', n_starts)
    _check_settings('local', 'likelihood', data, bounds, 1)
    if method == 'refits':
        return _get_refits(
            source,
            data,
            bounds,
            noise=tuple(float(item) for item in noise),
            n=n,
            dt=dt,
            solver=solver,
            random_state=random_state,
            executor=executor,
        )
    objective = _CalibrationObjective(
        source,
        _series_of(data),
        _check_priors(bounds),
        dt=dt,
        solver=solver,
        random_state=random_state,
    )
    return _get_profiles(objective, level, n_starts, random_state)
