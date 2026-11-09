"""Measure large deviations.

The methods are generating functions, rate functions, cloning, and exponential reweighting.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
from scipy import linalg, special

from ..components.systems import System
from ..engine.problems import ProblemSpec
from ..engine.runtime import Trajectory
from ..engine.runtime._simulations import Simulation
from ..errors import AnalysisError
from ..specs import AnalysisTags
from ..types import Array, DataKind
from ._base import _check_choice, _check_settings, _composite, _read_series
from .types import Source

MIN_SIZES = 2
SMALL_POPULATION = 0.1
MIN_EFFECTIVE_CLONES = 10
SETTINGS = {
    'tilted': {'weights', 'tilts', 'values'},
    'block': {'tilts', 'block', 'dt', 'values'},
    'cloning': {'observable', 'tilts', 'values', 'population', 'interval', 'until', 'random_state', 'simulation'},
    'histogram': {'bins', 'value_range'},
    'reweight': {'log_weights'},
}
"""Settings every kind reads, besides `kind` and `variables`."""

KINDS = ('tilted', 'block', 'cloning', 'histogram', 'reweight')

TAGS = AnalysisTags(DataKind.PATH)
KIND_TAGS = {
    'block': AnalysisTags(DataKind.SERIES),
    'histogram': AnalysisTags(DataKind.ENSEMBLE),
    'reweight': AnalysisTags(DataKind.ENSEMBLE),
}


@dataclass(frozen=True)
class LargeDeviationsResult:
    """Large deviations of a time-additive observable.

    They are its generating function and rate function.

    Args:
        rates:
            The rate matrix of `'tilted'`.

        increments:
            The increments of `'block'`.

        system:
            The system or its compiled problem of `'cloning'`.

        samples:
            The samples at every size of `'histogram'`.

        observations:
            The samples of `'reweight'`.

        tilts:
            The tilts of the generating function.

        scgf:
            The scaled cumulant generating function at every tilt.

        effective_population:
            The smallest effective number of clones over the selection steps at every tilt, `(Σ w)² / Σ w²`.

        values:
            The values of the rate function, the centres of the bins of `'histogram'`.

        rate_function:
            The rate function at every value, 0 at its minimum and `nan` where a size left the bin empty for
            `'histogram'`.

        weights:
            The normalized weight of every sample of `'reweight'`.

        mean:
            The weighted mean of the samples of `'reweight'`.

        effective_size:
            The effective number of samples `1 / Σ w²` of `'reweight'`.
    """

    rates: Array | None = None
    increments: Array | None = None
    system: System | ProblemSpec | None = None
    samples: dict[int, Array] | None = None
    observations: Array | None = None
    tilts: Array | None = None
    scgf: Array | None = None
    effective_population: Array | None = None
    values: Array | None = None
    rate_function: Array | None = None
    weights: Array | None = None
    mean: float | None = None
    effective_size: float | None = None


def large_deviations(
    source: Source | ProblemSpec | Mapping[int, npt.ArrayLike],
    *,
    kind: str = 'tilted',
    variables: Sequence[str] | None = None,
    weights: npt.ArrayLike | None = None,
    tilts: float | npt.ArrayLike = 1.0,
    values: npt.ArrayLike | None = None,
    block: int = 10,
    dt: float = 1.0,
    observable: str | Callable[[Trajectory], float] | None = None,
    population: int = 1000,
    interval: float = 1.0,
    until: float = 20.0,
    random_state: int | None = None,
    simulation: Mapping[str, object] | None = None,
    bins: int | npt.ArrayLike = 41,
    value_range: tuple[float, float] | None = None,
    log_weights: npt.ArrayLike | None = None,
) -> LargeDeviationsResult:
    """Return the large deviations of a time-additive observable, of one kind.

    The kinds estimate the scaled cumulant generating function `lambda(k)` at the `tilts`, or the rate function `I`:

    - `'tilted'`: exactly, for a counting observable of a jump process that adds `weights[i, j]` at every jump from
      `i` to `j`, as the largest eigenvalue of the tilted generator `rates * exp(k weights) - diag(escape rates)` of
      the rate matrix. A diagonal rate counts jumps that keep the state, such as the events of a Poisson process.
    - `'block'`: from blocks of `block` steps of one long run of the increments of the observable, as
      `ln <exp(k A)> / (block dt)`.
    - `'cloning'`: by the fixed-population cloning of Giardinà, Kurchan, and Peliti, which advances every clone of a
      system by `interval` through the simulation's steppers, weights it by `exp(k A)` for the increment `A` of the
      observable, and resamples the population by the weights. The system is compiled once, and every segment restarts
      the clock at zero, so the laws must not depend on time explicitly. A warning names a population whose effective
      size falls below a tenth of the clones or below 10 clones.
    - `'histogram'`: the rate function of `P_N(m) ≍ exp(-N I(m))` from the samples of an order parameter at several
      sizes `N`, fitting `-ln P_N` linearly in `N` in every bin.
    - `'reweight'`: samples reweighted to another law by the logarithms of the likelihood ratios, such as `beta dh M`.

    With `values`, the generating function kinds also give the rate function, its Legendre transform
    `I(a) = max_k (k a - lambda(k))` over the tilts.

    Args:
        source:
            The rate matrix for `'tilted'`, the trajectory or an array of increments for `'block'`, the system or its
            problem for `'cloning'`, a mapping of every size to its samples for `'histogram'`, or the samples for
            `'reweight'`.

        kind:
            The estimate, one of the kinds above.

        variables:
            The series of a trajectory whose values are the increments of `'block'`, or `None` for the first one.

        weights:
            The increment of the observable at every jump of `'tilted'`.

        tilts:
            The values of the tilt `k`.

        values:
            The values `a` of the observable per unit time of the rate function, or `None` for none.

        block:
            The number of steps of every block.

        dt:
            The time of one step.

        observable:
            The dotted name of a state whose increment over each interval is the observable, such as a counter, or a
            function of the trajectory of one interval that returns the increment.

        population:
            The number of clones.

        interval:
            The time between two selection steps.

        until:
            The simulated time of every clone, a multiple of the interval.

        random_state:
            The seed of the clones and of the selection.

        simulation:
            Other run settings, such as `solver` or `dt`.

        bins:
            The number of bins or their edges of `'histogram'`.

        value_range:
            The range of the bins, or `None` for the range of all samples.

        log_weights:
            The logarithm of the ratio of the new to the old probability of every sample.

    Returns:
        result:
            The data read, the generating function at every tilt, and the rate function at the values.

    Raises:
        AnalysisError:
            The kind is unknown, a setting the kind does not read is changed, a setting the kind needs is missing, a
            setting is out of range, or the source does not suit the kind.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import large_deviations
        >>> large_deviations([[2.0]], weights=[[1.0]], tilts=[0.0, 1.0]).scgf.round(4).tolist()
        [0.0, 3.4366]
        >>> tilts = np.linspace(-3, 3, 6001)
        >>> rate = large_deviations([[2.0]], weights=[[1.0]], tilts=tilts, values=[3.0])
        >>> round(float(rate.rate_function[0]), 4)
        0.2164
        >>> large_deviations([-1.0, 1.0], kind='reweight', log_weights=[0.0, 0.0]).mean
        0.0
    """
    source = _composite(source)
    _check_choice('large_deviations', 'kind', kind, KINDS)
    _check_settings(
        large_deviations,
        'kind',
        SETTINGS[kind] | {'variables'},
        kind=kind,
        variables=variables,
        weights=weights,
        tilts=tilts,
        values=values,
        block=block,
        dt=dt,
        observable=observable,
        population=population,
        interval=interval,
        until=until,
        random_state=random_state,
        simulation=simulation,
        bins=bins,
        value_range=value_range,
        log_weights=log_weights,
    )
    needed = {'tilted': weights, 'cloning': observable, 'reweight': log_weights}
    names = {'tilted': 'weights', 'cloning': 'observable', 'reweight': 'log_weights'}
    if kind in needed and needed[kind] is None:
        error_msg = f'large_deviations of kind {kind!r} needs {names[kind]}. Give it.'
        raise AnalysisError(error_msg)
    _check(population, interval, until, block, dt)
    if kind == 'histogram':
        if not isinstance(source, Mapping):
            error_msg = 'large_deviations of kind histogram reads a mapping of every size to its samples.'
            raise AnalysisError(error_msg)
        samples = {size: np.asarray(source[size], dtype=float) for size in sorted(source)}
        centres, histogram_rates = _histogram(samples, bins, value_range)
        return LargeDeviationsResult(samples=samples, values=centres, rate_function=histogram_rates)
    if kind == 'reweight':
        observations = np.asarray(source, dtype=float)
        logs = np.asarray(log_weights, dtype=float)
        normalized = np.exp(logs - special.logsumexp(logs))
        return LargeDeviationsResult(
            observations=observations,
            weights=normalized,
            mean=float(np.sum(normalized * observations)),
            effective_size=float(1 / np.sum(normalized**2)),
        )
    grid = np.atleast_1d(np.asarray(tilts, dtype=float))
    rates: Array | None = None
    increments: Array | None = None
    system: System | ProblemSpec | None = None
    effective_population: Array | None = None
    if kind == 'tilted':
        rates = np.asarray(source, dtype=float)
        tilted = np.asarray(weights, dtype=float)
        escape = np.diag(rates.sum(axis=1))
        scgf = np.asarray([linalg.eigvals(rates * np.exp(tilt * tilted) - escape).real.max() for tilt in grid])
    elif kind == 'block':
        increments = _read_series(source, variables).values[:, 0].astype(float)
        scgf = _block_scgf(increments, grid, block, dt)
    else:
        if not isinstance(source, System | ProblemSpec):
            error_msg = f'large_deviations of kind cloning reads a system or its problem, not {type(source).__name__}.'
            raise AnalysisError(error_msg)
        estimates = [
            _clone(
                source,
                float(tilt),
                observable=observable,
                population=population,
                interval=interval,
                until=until,
                random_state=random_state,
                simulation=simulation,
            )
            for tilt in grid
        ]
        scgf = np.asarray([value for value, _ in estimates])
        effective_population = np.asarray([effective for _, effective in estimates])
        system = source
    points: Array | None = None
    rate_function: Array | None = None
    if values is not None:
        points = np.atleast_1d(np.asarray(values, dtype=float))
        rate_function = _legendre_transform(grid, scgf, points)
    return LargeDeviationsResult(
        rates=rates,
        increments=increments,
        system=system,
        tilts=grid,
        scgf=scgf,
        effective_population=effective_population,
        values=points,
        rate_function=rate_function,
    )


def _block_scgf(array: Array, tilts: Array, block: int, dt: float) -> Array:
    """Return the generating function over the blocks of the run."""
    count = array.size // block
    if count < MIN_SIZES:
        error_msg = f'The run of {array.size} steps holds fewer than 2 blocks of {block} steps. Use shorter blocks.'
        raise AnalysisError(error_msg)
    sums = array[: count * block].reshape(count, block).sum(axis=1)
    return (special.logsumexp(np.outer(tilts, sums), axis=1) - np.log(count)) / (block * dt)


def _clone(
    system: System | ProblemSpec,
    tilt: float,
    *,
    observable: str | Callable[[Trajectory], float] | None,
    population: int,
    interval: float,
    until: float,
    random_state: int | None,
    simulation: Mapping[str, object] | None,
) -> tuple[float, float]:
    """Return the generating function at one tilt and the least effective population.

    They are found by cloning.
    """
    if observable is None:
        error_msg = 'large_deviations of method cloning needs the observable whose increments it tilts.'
        raise AnalysisError(error_msg)
    steps = round(until / interval)
    rng = np.random.default_rng(random_state)
    settings = {'dt': interval, **(simulation or {})}
    first = Simulation(system).set_params(**settings).prepare().run(until=interval)
    problem = first.problem_
    states = np.tile(problem.initial_state, (population, 1))
    log_growth, effective = 0.0, float(population)
    for _ in range(steps):
        increments = np.empty(population)
        for index in range(population):
            start = replace(problem, initial_state=states[index].copy())
            seed = int(rng.integers(2**31 - 1))
            run = Simulation(start, random_state=seed).set_params(**settings).prepare().run(until=interval)
            increments[index] = _increment(observable, run.trajectory_)
            states[index] = np.asarray(run.trajectory_.states)[-1]
        logs = tilt * increments
        log_growth += float(special.logsumexp(logs) - np.log(population))
        weights = np.exp(logs - special.logsumexp(logs))
        effective = min(effective, float(1 / np.sum(weights**2)))
        states = states[rng.choice(population, size=population, p=weights)]
    if effective < max(population * SMALL_POPULATION, MIN_EFFECTIVE_CLONES):
        warnings.warn(
            f'The effective population of the cloning fell to {effective:.1f} of {population} clones, so the estimate '
            'may be biased. Use more clones or a shorter interval.',
            RuntimeWarning,
            stacklevel=3,
        )
    return log_growth / until, effective


def _histogram(
    samples: dict[int, Array],
    bins: int | npt.ArrayLike,
    value_range: tuple[float, float] | None,
) -> tuple[Array, Array]:
    """Return the centres of the bins and the rate function.

    The rate function comes from the histograms of the sizes.
    """
    sizes = list(samples)
    if len(sizes) < MIN_SIZES:
        error_msg = f'A rate function from histograms needs at least {MIN_SIZES} sizes, not {len(sizes)}.'
        raise AnalysisError(error_msg)
    arrays = list(samples.values())
    edges = np.histogram_bin_edges(np.concatenate(arrays), bins=bins, range=value_range)
    densities = np.asarray([np.histogram(array, bins=edges, density=True)[0] for array in arrays])
    with np.errstate(divide='ignore'):
        logs = -np.log(densities)
    rates = np.full(edges.size - 1, np.nan)
    for index in range(rates.size):
        column = logs[:, index]
        finite = np.isfinite(column)
        if finite.sum() >= MIN_SIZES:
            rates[index] = np.polyfit(np.asarray(sizes, dtype=float)[finite], column[finite], 1)[0]
    return (edges[1:] + edges[:-1]) / 2, rates - np.nanmin(rates)


def _increment(observable: str | Callable[[Trajectory], float], trajectory: Trajectory) -> float:
    """Return the increment of a time-additive observable over one segment of a clone."""
    if callable(observable):
        return float(observable(trajectory))
    values = np.asarray(trajectory.values(observable), dtype=float)
    return float(np.sum(values[-1] - values[0]))


def _legendre_transform(tilts: Array, function: Array, values: Array) -> Array:
    """Return the rate function `I(a) = max_k (k a - lambda(k))`.

    The generating function is sampled on a grid.
    """
    return np.max(np.outer(values, tilts) - function[None, :], axis=1)


def _check(population: int, interval: float, until: float, block: int, dt: float) -> None:
    """Raise an error for a cloning schedule, a block, or a step out of range."""
    steps = round(until / interval)
    if population < MIN_SIZES or steps < 1 or not np.isclose(steps * interval, until):
        error_msg = (
            f'Cloning needs at least {MIN_SIZES} clones and a duration that is a positive multiple of the interval, '
            f'but got {population} clones, the duration {until}, and the interval {interval}.'
        )
        raise AnalysisError(error_msg)
    if block < 1 or dt <= 0:
        error_msg = f'large_deviations has block={block!r} and dt={dt!r}. Use positive values.'
        raise AnalysisError(error_msg)
