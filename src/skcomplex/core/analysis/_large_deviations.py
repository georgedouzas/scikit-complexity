"""Measure large deviations: generating functions, rate functions, cloning, and exponential reweighting."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt
from scipy import linalg, special

from ..components.system import System
from ..engine.problems import ProblemSpec
from ..engine.runtime import Simulation, Trajectory
from ..errors import AnalysisError
from ..types import Array, DataKind
from ._base import BaseAnalysis
from .types import Source

MIN_SIZES = 2
SMALL_POPULATION = 0.1
MIN_EFFECTIVE_CLONES = 10
KINDS = ('tilted', 'block', 'cloning', 'histogram', 'reweight')
SETTINGS = {
    'tilted': {'weights', 'tilts', 'values'},
    'block': {'tilts', 'block', 'dt', 'values'},
    'cloning': {'observable', 'tilts', 'values', 'population', 'interval', 'until', 'random_state', 'simulation'},
    'histogram': {'bins', 'value_range'},
    'reweight': {'log_weights'},
}
"""Settings every kind reads, besides `kind` and `variables`."""

DATA_KINDS = {
    'tilted': DataKind.PATH,
    'block': DataKind.SERIES,
    'cloning': DataKind.PATH,
    'histogram': DataKind.ENSEMBLE,
    'reweight': DataKind.ENSEMBLE,
}


class LargeDeviationAnalysis(BaseAnalysis):
    """Large deviations of a time-additive observable or of an order parameter, of one kind.

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
            Other settings of `Simulation`, such as `solver` or `dt`.

        bins:
            The number of bins or their edges of `'histogram'`.

        value_range:
            The range of the bins, or `None` for the range of all samples.

        log_weights:
            The logarithm of the ratio of the new to the old probability of every sample.

    Attributes:
        rates_:
            The rate matrix of `'tilted'`.

        increments_:
            The increments of `'block'`.

        system_:
            The system or its compiled problem of `'cloning'`.

        samples_:
            The samples at every size of `'histogram'`.

        observations_:
            The samples of `'reweight'`.

        tilts_:
            The tilts of the generating function.

        scgf_:
            The scaled cumulant generating function at every tilt.

        effective_population_:
            The smallest effective number of clones over the selection steps at every tilt, `(Σ w)² / Σ w²`.

        values_:
            The values of the rate function, the centres of the bins of `'histogram'`.

        rate_function_:
            The rate function at every value, 0 at its minimum and `nan` where a size left the bin empty for
            `'histogram'`.

        weights_:
            The normalized weight of every sample of `'reweight'`.

        mean_:
            The weighted mean of the samples of `'reweight'`.

        effective_size_:
            The effective number of samples `1 / Σ w²` of `'reweight'`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import LargeDeviationAnalysis
        >>> tilted = LargeDeviationAnalysis(weights=[[1.0]], tilts=[0.0, 1.0])
        >>> tilted.prepare([[2.0]]).run().scgf_.round(4).tolist()
        [0.0, 3.4366]
        >>> tilts = np.linspace(-3, 3, 6001)
        >>> rate = LargeDeviationAnalysis(weights=[[1.0]], tilts=tilts, values=[3.0])
        >>> round(float(rate.prepare([[2.0]]).run().rate_function_[0]), 4)
        0.2164
        >>> LargeDeviationAnalysis('reweight', log_weights=[0.0, 0.0]).prepare([-1.0, 1.0]).run().mean_
        0.0
    """

    def __init__(
        self,
        kind: str = 'tilted',
        variables: Sequence[str] | None = None,
        *,
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
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.weights = weights
        self.tilts = tilts
        self.values = values
        self.block = block
        self.dt = dt
        self.observable = observable
        self.population = population
        self.interval = interval
        self.until = until
        self.random_state = random_state
        self.simulation = simulation
        self.bins = bins
        self.value_range = value_range
        self.log_weights = log_weights

    def _check_needed(self, name: str) -> None:
        """Raise an error for a setting the kind needs that is not given."""
        if getattr(self, name) is None:
            error_msg = f'{type(self).__name__} of kind {self.kind!r} needs {name}. Give it.'
            raise AnalysisError(error_msg)

    def _check(self) -> None:
        """Validate the kind, the settings each kind needs, and the cloning schedule."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_settings('kind', SETTINGS[self.kind] | {'variables'})
        needed = {'tilted': 'weights', 'cloning': 'observable', 'reweight': 'log_weights'}
        if self.kind in needed:
            self._check_needed(needed[self.kind])
        steps = round(self.until / self.interval)
        if self.population < MIN_SIZES or steps < 1 or not np.isclose(steps * self.interval, self.until):
            error_msg = (
                f'Cloning needs at least {MIN_SIZES} clones and a duration that is a positive multiple of the '
                f'interval, but got {self.population} clones, the duration {self.until}, and the interval '
                f'{self.interval}.'
            )
            raise AnalysisError(error_msg)
        if self.block < 1 or self.dt <= 0:
            error_msg = f'{type(self).__name__} has block={self.block!r} and dt={self.dt!r}. Use positive values.'
            raise AnalysisError(error_msg)

    def _get_data_kind(self) -> DataKind:
        """Return the path kind for jump processes, the series kind for blocks, and the ensemble kind for samples."""
        return DATA_KINDS[self.kind]

    def _prepare(self, source: Source | ProblemSpec | Mapping[int, npt.ArrayLike]) -> None:
        """Read the rate matrix, the increments, the system, or the samples the kind estimates from."""
        if self.kind == 'tilted':
            self.rates_ = np.asarray(source, dtype=float)
        elif self.kind == 'block':
            self.increments_ = self._read_series(source, self.variables).values[:, 0].astype(float)
        elif self.kind == 'cloning':
            if not isinstance(source, System | ProblemSpec):
                error_msg = (
                    f'{type(self).__name__} of kind cloning reads a system or its problem, not {type(source).__name__}.'
                )
                raise AnalysisError(error_msg)
            self.system_ = source
        elif self.kind == 'histogram':
            if not isinstance(source, Mapping):
                error_msg = f'{type(self).__name__} of kind histogram reads a mapping of every size to its samples.'
                raise AnalysisError(error_msg)
            self.samples_ = {size: np.asarray(source[size], dtype=float) for size in sorted(source)}
        else:
            self.observations_ = np.asarray(source, dtype=float)

    def _run_tilted(self) -> None:
        """Compute the largest eigenvalue of the tilted generator at every tilt."""
        increments = np.asarray(self.weights, dtype=float)
        escape = np.diag(self.rates_.sum(axis=1))
        self.scgf_ = np.asarray(
            [linalg.eigvals(self.rates_ * np.exp(tilt * increments) - escape).real.max() for tilt in self.tilts_],
        )

    def _run_block(self) -> None:
        """Compute the generating function over the blocks of the run."""
        array = self.increments_
        count = array.size // self.block
        if count < MIN_SIZES:
            error_msg = (
                f'The run of {array.size} steps holds fewer than 2 blocks of {self.block} steps. Use shorter blocks.'
            )
            raise AnalysisError(error_msg)
        sums = array[: count * self.block].reshape(count, self.block).sum(axis=1)
        self.scgf_ = (special.logsumexp(np.outer(self.tilts_, sums), axis=1) - np.log(count)) / (self.block * self.dt)

    def _clone(self, tilt: float) -> tuple[float, float]:
        """Return the generating function at one tilt and the smallest effective population, by cloning."""
        observable = self.observable
        if observable is None:
            error_msg = f'{type(self).__name__} of method cloning needs the observable whose increments it tilts.'
            raise AnalysisError(error_msg)
        population, steps = self.population, round(self.until / self.interval)
        rng = np.random.default_rng(self.random_state)
        settings = {'dt': self.interval, **(self.simulation or {})}
        simulation = Simulation(self.system_).set_params(**settings).prepare().run(until=self.interval)
        problem = simulation.problem_
        states = np.tile(problem.initial_state, (population, 1))
        log_growth, effective = 0.0, float(population)
        for _ in range(steps):
            increments = np.empty(population)
            for index in range(population):
                start = replace(problem, initial_state=states[index].copy())
                seed = int(rng.integers(2**31 - 1))
                simulation = (
                    Simulation(start, random_state=seed).set_params(**settings).prepare().run(until=self.interval)
                )
                increments[index] = self._increment(observable, simulation.trajectory_)
                states[index] = np.asarray(simulation.trajectory_.states)[-1]
            logs = tilt * increments
            log_growth += float(special.logsumexp(logs) - np.log(population))
            weights = np.exp(logs - special.logsumexp(logs))
            effective = min(effective, float(1 / np.sum(weights**2)))
            states = states[rng.choice(population, size=population, p=weights)]
        if effective < max(population * SMALL_POPULATION, MIN_EFFECTIVE_CLONES):
            warnings.warn(
                f'The effective population of the cloning fell to {effective:.1f} of {population} clones, so the '
                'estimate may be biased. Use more clones or a shorter interval.',
                RuntimeWarning,
                stacklevel=2,
            )
        return log_growth / self.until, effective

    def _run_histogram(self) -> None:
        """Compute the rate function from the histograms of the sizes."""
        sizes = list(self.samples_)
        if len(sizes) < MIN_SIZES:
            error_msg = f'A rate function from histograms needs at least {MIN_SIZES} sizes, not {len(sizes)}.'
            raise AnalysisError(error_msg)
        arrays = list(self.samples_.values())
        edges = np.histogram_bin_edges(np.concatenate(arrays), bins=self.bins, range=self.value_range)
        densities = np.asarray([np.histogram(array, bins=edges, density=True)[0] for array in arrays])
        with np.errstate(divide='ignore'):
            logs = -np.log(densities)
        rates = np.full(edges.size - 1, np.nan)
        for index in range(rates.size):
            column = logs[:, index]
            finite = np.isfinite(column)
            if finite.sum() >= MIN_SIZES:
                rates[index] = np.polyfit(np.asarray(sizes, dtype=float)[finite], column[finite], 1)[0]
        self.values_ = (edges[1:] + edges[:-1]) / 2
        self.rate_function_ = rates - np.nanmin(rates)

    def _run_reweight(self) -> None:
        """Compute the weights of the samples and the reweighted mean."""
        logs = np.asarray(self.log_weights, dtype=float)
        self.weights_ = np.exp(logs - special.logsumexp(logs))
        self.mean_ = float(np.sum(self.weights_ * self.observations_))
        self.effective_size_ = float(1 / np.sum(self.weights_**2))

    def _run(self, inputs: object = None) -> None:
        """Compute the estimate of the kind, and the rate function at the values."""
        if self.kind == 'histogram':
            self._run_histogram()
            return
        if self.kind == 'reweight':
            self._run_reweight()
            return
        self.tilts_ = np.atleast_1d(np.asarray(self.tilts, dtype=float))
        if self.kind == 'tilted':
            self._run_tilted()
        elif self.kind == 'block':
            self._run_block()
        else:
            estimates = [self._clone(float(tilt)) for tilt in self.tilts_]
            self.scgf_ = np.asarray([scgf for scgf, _ in estimates])
            self.effective_population_ = np.asarray([effective for _, effective in estimates])
        if self.values is not None:
            self.values_ = np.atleast_1d(np.asarray(self.values, dtype=float))
            self.rate_function_ = self._legendre_transform(self.tilts_, self.scgf_, self.values_)

    @staticmethod
    def _increment(observable: str | Callable[[Trajectory], float], trajectory: Trajectory) -> float:
        """Return the increment of a time-additive observable over one segment of a clone."""
        if callable(observable):
            return float(observable(trajectory))
        values = np.asarray(trajectory.values(observable), dtype=float)
        return float(np.sum(values[-1] - values[0]))

    @staticmethod
    def _legendre_transform(tilts: Array, function: Array, values: Array) -> Array:
        """Return the rate function `I(a) = max_k (k a - lambda(k))` of a generating function sampled on a grid."""
        return np.max(np.outer(values, tilts) - function[None, :], axis=1)
