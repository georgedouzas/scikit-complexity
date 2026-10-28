"""Measure the entropy of series, symbols, and fields: of distributions, blocks, ordinal patterns, and regularity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
import math
from collections import Counter
from collections.abc import Hashable, Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
import ordpy
from scipy import stats

from ..core.errors import AnalysisError
from ._base import BaseAnalysis, Source
from ._kinds import DataKind
from ._lazy import module

type Array = npt.NDArray[np.float64]
KINDS = (
    'shannon',
    'renyi',
    'tsallis',
    'block',
    'rate',
    'excess',
    'lempel-ziv',
    'permutation',
    'sample',
    'approximate',
    'multiscale',
    'spectral',
    'ensemble',
)
MEASURES = ('shannon', 'renyi', 'tsallis')
TIES = ('order', 'random')
SETTINGS = {
    'shannon': {'base', 'probabilities'},
    'renyi': {'order', 'base', 'probabilities'},
    'tsallis': {'order', 'probabilities'},
    'block': {'dimension', 'base'},
    'rate': {'dimension', 'base'},
    'excess': {'dimension', 'base'},
    'lempel-ziv': {'normalize'},
    'permutation': {'dimension', 'delay', 'normalize', 'measure', 'order', 'ties', 'random_state'},
    'sample': {'dimension', 'tolerance'},
    'approximate': {'dimension', 'tolerance'},
    'multiscale': {'dimension', 'tolerance', 'scales'},
    'spectral': {'dt', 'normalize'},
    'ensemble': {'reference'},
}
"""Settings every kind reads, besides `kind` and `variables`."""


def _probabilities(items: Sequence[Hashable]) -> Array:
    """Return the empirical probabilities of distinct items."""
    counts = np.asarray(list(Counter(items).values()), dtype=float)
    return counts / counts.sum()


def _distribution(values: npt.ArrayLike, *, probabilities: bool) -> Array:
    """Return the nonzero probabilities of symbols, or of a probability vector."""
    p = np.asarray(values, dtype=float) if probabilities else _probabilities(list(np.asarray(values).ravel()))
    return p[p > 0]


def _shannon(values: npt.ArrayLike, *, base: float = 2.0, probabilities: bool = False) -> float:
    """Return the Shannon entropy of symbols, or of a probability vector."""
    return float(stats.entropy(_distribution(values, probabilities=probabilities), base=base))


def _renyi(values: npt.ArrayLike, order: float, *, base: float = 2.0, probabilities: bool = False) -> float:
    """Return the Rényi entropy `log(Σ p^a) / (1 - a)`, the Shannon entropy at order 1 and `-log max p` at `inf`."""
    p = _distribution(values, probabilities=probabilities)
    if order == 1:
        return _shannon(p, base=base, probabilities=True)
    if math.isinf(order):
        return float(-np.log(p.max()) / np.log(base))
    return float(np.log(np.sum(p**order)) / (1 - order) / np.log(base))


def _tsallis(values: npt.ArrayLike, index: float, *, probabilities: bool = False) -> float:
    """Return the Tsallis entropy `(1 - Σ p^q) / (q - 1)`, the Shannon entropy in nats at index 1."""
    p = _distribution(values, probabilities=probabilities)
    if index == 1:
        return _shannon(p, base=math.e, probabilities=True)
    return float((1 - np.sum(p**index)) / (index - 1))


def _block_entropy(symbols: npt.ArrayLike, m: int, *, base: float = 2.0) -> float:
    """Return the entropy of the blocks of `m` consecutive symbols, or of square patches of side `m` of a lattice."""
    array = np.asarray(symbols)
    if m == 0:
        return 0.0
    windows = np.lib.stride_tricks.sliding_window_view(array, (m,) * array.ndim)
    blocks = windows.reshape(-1, m**array.ndim)
    return _shannon(_probabilities([block.tobytes() for block in blocks]), base=base, probabilities=True)


def _block_entropies(symbols: npt.ArrayLike, max_m: int, *, base: float = 2.0) -> Array:
    """Return the block entropies `S_0` to `S_max_m`."""
    return np.asarray([_block_entropy(symbols, m, base=base) for m in range(max_m + 1)])


def _entropy_rate(symbols: npt.ArrayLike, max_m: int, *, base: float = 2.0) -> float:
    """Return the entropy rate, the limit of `S_m - S_{m-1}`, estimated at the largest block length."""
    entropies = _block_entropies(symbols, max_m, base=base)
    return float(max(entropies[-1] - entropies[-2], 0.0))


def _excess_entropy(symbols: npt.ArrayLike, max_m: int, *, base: float = 2.0) -> float:
    """Return the excess entropy `S_m - m s` at the largest block length."""
    entropies = _block_entropies(symbols, max_m, base=base)
    return float(entropies[-1] - max_m * max(entropies[-1] - entropies[-2], 0.0))


def _break_ties(values: npt.ArrayLike, ties: str, random_state: int | None) -> Array:
    """Return values with ties kept in order, or broken by random offsets below a quarter of the smallest gap."""
    array = np.asarray(values, dtype=float)
    if ties == 'order':
        return array
    gaps = np.diff(np.unique(array))
    scale = gaps.min() / 4 if gaps.size else 1.0
    return array + np.random.default_rng(random_state).uniform(0, scale, array.shape)


def _measure_of(probabilities: Array, measure: str, order: float) -> float:
    """Return the Shannon or Rényi entropy in bits, or the Tsallis entropy, of probabilities."""
    if measure == 'renyi':
        return _renyi(probabilities, order, probabilities=True)
    if measure == 'tsallis':
        return _tsallis(probabilities, order, probabilities=True)
    return _shannon(probabilities, probabilities=True)


def _largest_measure(outcomes: int, measure: str, order: float) -> float:
    """Return the entropy of the uniform law on a number of outcomes."""
    if measure == 'tsallis':
        return _tsallis(np.full(outcomes, 1 / outcomes), order, probabilities=True)
    return math.log2(outcomes)


def _tolerance(array: Array, tolerance: float) -> float:
    """Return the tolerance as a fraction of the standard deviation of a series."""
    return float(tolerance * array.std())


def _sample(values: npt.ArrayLike, dimension: int, tolerance: float) -> float:
    """Return the sample entropy `-ln(A / B)` of a series, with the tolerance a fraction of its deviation."""
    array = np.ascontiguousarray(values, dtype=float)
    radius = _tolerance(array, tolerance)
    return float(module('antropy').sample_entropy(array, order=dimension, tolerance=radius)) + 0.0


class EntropyAnalysis(BaseAnalysis):
    """Entropy of every series of a source, or of every frame of a field, of one kind.

    The kinds read symbols, patterns, or real values:

    - `'shannon'`, `'renyi'`, and `'tsallis'`: the entropy of the distribution of the symbols, or of a probability
      vector with `probabilities=True`, the Rényi order and the Tsallis index given by `order`.
    - `'block'`, `'rate'`, and `'excess'`: the block entropies `S_0` to `S_m`, the entropy rate, and the excess
      entropy of the blocks of up to `dimension` symbols, or of square patches of a field.
    - `'lempel-ziv'`: the number of new phrases of the LZ76 parsing.
    - `'permutation'`: the entropy of the ordinal patterns of length `dimension`, or of the windows
      `(columns, rows)` of a field, measured with `measure`.
    - `'sample'` and `'approximate'`: the regularity of patterns of length `dimension`, close within `tolerance`
      standard deviations.
    - `'multiscale'`: the sample entropy of the means of non-overlapping windows of every scale of `scales`.
    - `'spectral'`: the Shannon entropy of the normalized periodogram.
    - `'ensemble'`: the Shannon entropy in bits of the outcomes of an ensemble at every frame, and their
      Kullback-Leibler divergence from the `reference` law, as in the H-theorem. The ensemble has one row per run and
      one column per frame of integer outcomes, such as `ExperimentResult.ensemble` returns.

    Args:
        kind:
            The entropy to compute, one of the kinds above.

        variables:
            The variables of a trajectory to read, or `None` for every scalar series, or the first field for a field.

        order:
            The order of the Rényi entropy or the index of the Tsallis entropy.

        base:
            The base of the logarithm, 2 for bits.

        probabilities:
            Whether every series is already a probability vector.

        dimension:
            The length of patterns or the largest block length, or the window `(columns, rows)` of a field.

        delay:
            The number of steps between the values of an ordinal pattern.

        tolerance:
            The largest distance of close patterns, as a fraction of the standard deviation.

        scales:
            The window lengths of the multiscale entropy.

        normalize:
            Whether to divide by the largest value, so the entropy lies between 0 and 1.

        dt:
            The time between samples of the spectral entropy.

        measure:
            The entropy of the ordinal patterns: `'shannon'`, `'renyi'`, or `'tsallis'`.

        ties:
            `'order'` to rank equal values by position, or `'random'` to rank them at random.

        random_state:
            The seed of the random ranks of equal values.

        reference:
            The probability of every outcome under the reference law of an ensemble, such as the stationary law.

    Attributes:
        values_:
            The series, one per column, or the frames of the field.

        labels_:
            The label of every series, or of the field.

        entropy_:
            One entropy per series or frame, with one column per block length for `'block'` and per scale for
            `'multiscale'`.

        divergence_:
            The divergence in bits of the outcomes of every frame of an ensemble from the reference law.

        error_:
            The expected divergence of a sample of the reference law of the ensemble's size, `(k - 1) / (2 n ln 2)`
            for `k` outcomes and `n` runs, below which a divergence is sampling noise.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import EntropyAnalysis
        >>> EntropyAnalysis().check().prepare([0, 1, 0, 1]).run().entropy_.tolist()
        [1.0]
        >>> EntropyAnalysis('excess', dimension=4).check().prepare([0, 0, 1, 1] * 100).run().entropy_.round(3).tolist()
        [2.0]
    """

    def __init__(
        self,
        kind: str = 'shannon',
        variables: Sequence[str] | None = None,
        *,
        order: float = 2.0,
        base: float = 2.0,
        probabilities: bool = False,
        dimension: int | tuple[int, int] = 3,
        delay: int = 1,
        tolerance: float = 0.2,
        scales: Sequence[int] = tuple(range(1, 11)),
        normalize: bool = True,
        dt: float = 1.0,
        measure: str = 'shannon',
        ties: str = 'order',
        random_state: int | None = None,
        reference: npt.ArrayLike | None = None,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.order = order
        self.base = base
        self.probabilities = probabilities
        self.dimension = dimension
        self.delay = delay
        self.tolerance = tolerance
        self.scales = scales
        self.normalize = normalize
        self.dt = dt
        self.measure = measure
        self.ties = ties
        self.random_state = random_state
        self.reference = reference

    def _check_choice(self, name: str, value: str, choices: tuple[str, ...]) -> None:
        """Raise an error for a setting that is not one of its choices."""
        if value not in choices:
            error_msg = f'{type(self).__name__} has {name}={value!r}. Use one of {", ".join(choices)}.'
            raise AnalysisError(error_msg)

    def _check_ignored(self) -> None:
        """Raise an error for a setting changed from its default that the kind does not read."""
        defaults = {
            name: parameter.default for name, parameter in inspect.signature(type(self).__init__).parameters.items()
        }
        read = SETTINGS[self.kind] | {'kind', 'variables'}
        ignored = [name for name, value in self.get_params().items() if name not in read and value != defaults[name]]
        if ignored:
            error_msg = (
                f'{type(self).__name__} of kind {self.kind!r} does not read {", ".join(ignored)}. It reads '
                f'{", ".join(sorted(SETTINGS[self.kind]))}.'
            )
            raise AnalysisError(error_msg)

    def _check(self) -> Self:
        """Validate the kind, the choices, the numeric settings, and that the kind reads every changed setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_choice('measure', self.measure, MEASURES)
        self._check_choice('ties', self.ties, TIES)
        self._check_ignored()
        if isinstance(self.dimension, tuple) and self.kind != 'permutation':
            error_msg = (
                f'{type(self).__name__} has the window dimension={self.dimension!r}, which only the permutation kind '
                'reads. Use one length.'
            )
            raise AnalysisError(error_msg)
        sizes = self.dimension if isinstance(self.dimension, tuple) else (self.dimension,)
        if any(int(size) < 1 for size in sizes) or self.delay < 1:
            error_msg = (
                f'{type(self).__name__} has dimension={self.dimension!r} and delay={self.delay!r}. Use 1 or more.'
            )
            raise AnalysisError(error_msg)
        if self.kind == 'ensemble' and self.reference is None:
            error_msg = f'{type(self).__name__} of kind ensemble needs the reference law. Set `reference`.'
            raise AnalysisError(error_msg)
        if self.tolerance <= 0 or self.base <= 0 or self.dt <= 0:
            error_msg = (
                f'{type(self).__name__} has tolerance={self.tolerance!r}, base={self.base!r}, and dt={self.dt!r}. '
                'Use positive values.'
            )
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the ensemble kind for ensembles, the field kind for windows of a field, and the series kind else."""
        if self.kind == 'ensemble':
            return DataKind.ENSEMBLE
        return DataKind.FIELD if isinstance(self.dimension, tuple) else DataKind.SERIES

    def _prepare(self, source: Source) -> Self:
        """Read the series of the source as columns, the frames of a field for windows, or the runs of an ensemble."""
        if self.kind == 'ensemble':
            self.values_ = np.asarray(source, dtype=np.int64)
            self.labels_ = ['ensemble']
            law = np.asarray(self.reference, dtype=float)
            if self.values_.min() < 0 or self.values_.max() >= law.size:
                error_msg = f'Outcomes must lie between 0 and {law.size - 1}, the outcomes of the reference law.'
                raise AnalysisError(error_msg)
            return self
        if self._get_data_kind() == DataKind.FIELD:
            name = self.variables[0] if self.variables else None
            frames = self._read_fields(source, name)
            self.values_ = frames[None] if frames.ndim == 2 else frames  # noqa: PLR2004
            self.labels_ = [name or 'field']
            return self
        series = self._read_series(source, self.variables)
        self.values_ = series.values
        self.labels_ = series.labels
        return self

    def _compute_distribution(self, values: npt.NDArray[np.generic]) -> float:
        """Return the Shannon, Rényi, or Tsallis entropy of symbols or of a probability vector."""
        if self.kind == 'renyi':
            return _renyi(values, self.order, base=self.base, probabilities=self.probabilities)
        if self.kind == 'tsallis':
            return _tsallis(values, self.order, probabilities=self.probabilities)
        return _shannon(values, base=self.base, probabilities=self.probabilities)

    def _compute(self, values: npt.NDArray[np.generic]) -> float | Array:
        """Return the entropy of one series or one frame."""
        kind = self.kind
        if kind in MEASURES:
            return self._compute_distribution(values)
        if kind in {'block', 'rate', 'excess'}:
            return self._compute_blocks(values)
        if kind == 'lempel-ziv':
            _, codes = np.unique(np.asarray(values).ravel(), return_inverse=True)
            return float(module('antropy').lziv_complexity(codes.astype(np.int64), normalize=self.normalize))
        if kind == 'permutation':
            return self._compute_permutation(values)
        return self._compute_regularity(np.ascontiguousarray(values, dtype=float))

    def _compute_blocks(self, symbols: npt.NDArray[np.generic]) -> float | Array:
        """Return the block entropies, the entropy rate, or the excess entropy of symbols."""
        max_m = int(self.dimension)  # type: ignore[arg-type]
        if self.kind == 'block':
            return _block_entropies(symbols, max_m, base=self.base)
        if self.kind == 'rate':
            return _entropy_rate(symbols, max_m, base=self.base)
        return _excess_entropy(symbols, max_m, base=self.base)

    def _compute_permutation(self, values: npt.NDArray[np.generic]) -> float:
        """Return the entropy of the ordinal patterns of a series or of the windows of a field."""
        dx, dy = (self.dimension, 1) if isinstance(self.dimension, int) else self.dimension
        array = _break_ties(values, self.ties, self.random_state)
        _, probabilities = ordpy.ordinal_distribution(array, dx=dx, dy=dy, taux=self.delay, tauy=self.delay)
        value = _measure_of(probabilities, self.measure, self.order)
        if self.normalize:
            value /= _largest_measure(math.factorial(dx * dy), self.measure, self.order)
        return float(abs(value))

    def _compute_regularity(self, array: Array) -> float | Array:
        """Return the sample, approximate, multiscale, or spectral entropy of a real series."""
        antropy = module('antropy')
        dimension = int(self.dimension)  # type: ignore[arg-type]
        if self.kind == 'sample':
            return _sample(array, dimension, self.tolerance)
        if self.kind == 'approximate':
            return float(antropy.app_entropy(array, order=dimension, tolerance=_tolerance(array, self.tolerance)))
        if self.kind == 'multiscale':
            radius = _tolerance(array, self.tolerance)
            entropies = []
            for scale in self.scales:
                count = array.size // scale
                coarse = np.ascontiguousarray(array[: count * scale].reshape(count, scale).mean(axis=1))
                entropies.append(antropy.sample_entropy(coarse, order=dimension, tolerance=radius))
            return np.asarray(entropies, dtype=float)
        return float(antropy.spectral_entropy(array, sf=1 / self.dt, method='fft', normalize=self.normalize))

    def _run_ensemble(self) -> Self:
        """Compute the entropy of every frame of an ensemble and its divergence from the reference law."""
        law = np.asarray(self.reference, dtype=float)
        histograms = [np.bincount(frame, minlength=law.size) / frame.size for frame in self.values_.T]
        self.entropy_ = np.asarray([_shannon(histogram, probabilities=True) for histogram in histograms])
        self.divergence_ = np.asarray([stats.entropy(histogram, law, base=2) for histogram in histograms])
        self.error_ = float((np.count_nonzero(law) - 1) / (2 * self.values_.shape[0] * np.log(2)))
        return self

    def _run(self) -> Self:
        """Compute the entropy of every series or frame, or of every frame of an ensemble."""
        if self.kind == 'ensemble':
            return self._run_ensemble()
        items = self.values_ if self._get_data_kind() == DataKind.FIELD else self.values_.T
        self.entropy_ = np.asarray([self._compute(item) for item in items], dtype=float)
        return self
