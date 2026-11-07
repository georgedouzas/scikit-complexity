"""Measure the entropy of series, symbols, and fields: of distributions, blocks, ordinal patterns, and regularity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import ordpy
from scipy import stats

from ..errors import AnalysisError
from ..specs import AnalysisTags, BaseSpec
from ..types import Array, DataKind
from ._base import SNAPSHOT_DIMENSIONS, _check_choice, _check_settings, _import, _read_fields, _read_series
from .types import Source

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

TIES = ('order', 'random')

TAGS = AnalysisTags(DataKind.SERIES)
KIND_TAGS = {
    'ensemble': AnalysisTags(DataKind.ENSEMBLE),
}


@dataclass(frozen=True)
class EntropyResult:
    """Entropy of every series of a source, of every frame of a field, or of every frame of an ensemble.

    Args:
        values:
            The series, one per column, or the frames of the field, or the runs of the ensemble.

        labels:
            The label of every series, or of the field.

        entropy:
            One entropy per series or frame, with one column per block length for `'block'` and per scale for
            `'multiscale'`.

        divergence:
            The divergence in bits of the outcomes of every frame of an ensemble from the reference law, or `None`
            for kinds other than `'ensemble'`.

        error:
            The expected divergence of a sample of the reference law of the ensemble's size, `(k - 1) / (2 n ln 2)`
            for `k` outcomes and `n` runs, below which a divergence is sampling noise, or `None` for kinds other than
            `'ensemble'`.
    """

    values: npt.NDArray[np.int64] | Array
    labels: list[str]
    entropy: Array
    divergence: Array | None = None
    error: float | None = None


def entropy(
    source: Source,
    *,
    kind: str = 'shannon',
    variables: Sequence[str] | None = None,
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
) -> EntropyResult:
    """Return the entropy of every series of a source, or of every frame of a field, of one kind.

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
        source:
            The trajectory, an array with one row per time and one column per series, the frames of a field, or the
            outcomes of an ensemble.

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

    Returns:
        result:
            The values read, their entropy, and the divergence of an ensemble.

    Raises:
        AnalysisError:
            A choice is unknown, a setting the kind does not read is changed, a setting is out of range, or the
            outcomes of an ensemble lie outside the reference law.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import entropy
        >>> entropy([0, 1, 0, 1]).entropy.tolist()
        [1.0]
        >>> entropy([0, 0, 1, 1] * 100, kind='excess', dimension=4).entropy.round(3).tolist()
        [2.0]
    """
    _check_choice('entropy', 'kind', kind, KINDS)
    _check_choice('entropy', 'measure', measure, MEASURES)
    _check_choice('entropy', 'ties', ties, TIES)
    _check_settings(
        entropy,
        'kind',
        SETTINGS[kind] | {'variables'},
        kind=kind,
        variables=variables,
        order=order,
        base=base,
        probabilities=probabilities,
        dimension=dimension,
        delay=delay,
        tolerance=tolerance,
        scales=scales,
        normalize=normalize,
        dt=dt,
        measure=measure,
        ties=ties,
        random_state=random_state,
        reference=reference,
    )
    if isinstance(dimension, tuple) and kind != 'permutation':
        error_msg = (
            f'entropy has the window dimension={dimension!r}, which only the permutation kind reads. Use one length.'
        )
        raise AnalysisError(error_msg)
    sizes = dimension if isinstance(dimension, tuple) else (dimension,)
    if any(int(size) < 1 for size in sizes) or delay < 1:
        error_msg = f'entropy has dimension={dimension!r} and delay={delay!r}. Use 1 or more.'
        raise AnalysisError(error_msg)
    if kind == 'ensemble' and reference is None:
        error_msg = 'entropy of kind ensemble needs the reference law. Set `reference`.'
        raise AnalysisError(error_msg)
    if tolerance <= 0 or base <= 0 or dt <= 0:
        error_msg = f'entropy has tolerance={tolerance!r}, base={base!r}, and dt={dt!r}. Use positive values.'
        raise AnalysisError(error_msg)
    if kind == 'ensemble':
        return _ensemble_entropy(source, np.asarray(reference, dtype=float))
    if isinstance(dimension, tuple):
        name = variables[0] if variables else None
        frames = _read_fields(source, name)
        values = frames[None] if frames.ndim == SNAPSHOT_DIMENSIONS else frames
        labels = [name or 'field']
        items = list(values)
    else:
        series = _read_series(source, variables)
        values, labels = series.values, series.labels
        items = list(values.T)
    settings = _EntropySpec(
        kind,
        order,
        base,
        probabilities,
        dimension,
        delay,
        tolerance,
        scales,
        normalize,
        dt,
        measure,
        ties,
        random_state,
    )
    result = np.asarray([_compute(item, settings) for item in items], dtype=float)
    return EntropyResult(values, labels, result)


@dataclass(frozen=True)
class _EntropySpec(BaseSpec):
    """Settings of one call of `entropy` that the computation of one series or frame reads."""

    kind: str
    order: float
    base: float
    probabilities: bool
    dimension: int | tuple[int, int]
    delay: int
    tolerance: float
    scales: Sequence[int]
    normalize: bool
    dt: float
    measure: str
    ties: str
    random_state: int | None


def _ensemble_entropy(source: Source, law: Array) -> EntropyResult:
    """Return the entropy of every frame of an ensemble and its divergence from the reference law."""
    values = np.asarray(source, dtype=np.int64)
    if values.min() < 0 or values.max() >= law.size:
        error_msg = f'Outcomes must lie between 0 and {law.size - 1}, the outcomes of the reference law.'
        raise AnalysisError(error_msg)
    histograms = [np.bincount(frame, minlength=law.size) / frame.size for frame in values.T]
    return EntropyResult(
        values,
        ['ensemble'],
        np.asarray([_shannon(histogram, probabilities=True) for histogram in histograms]),
        np.asarray([stats.entropy(histogram, law, base=2) for histogram in histograms]),
        float((np.count_nonzero(law) - 1) / (2 * values.shape[0] * np.log(2))),
    )


def _compute(values: npt.NDArray[np.generic], settings: _EntropySpec) -> float | Array:
    """Return the entropy of one series or one frame."""
    kind = settings.kind
    if kind in MEASURES:
        return _compute_distribution(values, settings)
    if kind in {'block', 'rate', 'excess'}:
        return _compute_blocks(values, settings)
    if kind == 'lempel-ziv':
        _, codes = np.unique(np.asarray(values).ravel(), return_inverse=True)
        return float(_import('antropy').lziv_complexity(codes.astype(np.int64), normalize=settings.normalize))
    if kind == 'permutation':
        return _compute_permutation(values, settings)
    return _compute_regularity(np.ascontiguousarray(values, dtype=float), settings)


def _compute_distribution(values: npt.NDArray[np.generic], settings: _EntropySpec) -> float:
    """Return the Shannon, Rényi, or Tsallis entropy of symbols or of a probability vector."""
    if settings.kind == 'renyi':
        return _renyi(values, settings.order, base=settings.base, probabilities=settings.probabilities)
    if settings.kind == 'tsallis':
        return _tsallis(values, settings.order, probabilities=settings.probabilities)
    return _shannon(values, base=settings.base, probabilities=settings.probabilities)


def _get_length(settings: _EntropySpec) -> int:
    """Return the length of patterns or the largest block length, raising an error for the window of a field."""
    if not isinstance(settings.dimension, int):
        error_msg = f'entropy of kind {settings.kind!r} takes one length as dimension, not a window.'
        raise AnalysisError(error_msg)
    return settings.dimension


def _compute_blocks(symbols: npt.NDArray[np.generic], settings: _EntropySpec) -> float | Array:
    """Return the block entropies, the entropy rate, or the excess entropy of symbols."""
    max_m = _get_length(settings)
    if settings.kind == 'block':
        return _block_entropies(symbols, max_m, base=settings.base)
    if settings.kind == 'rate':
        return _entropy_rate(symbols, max_m, base=settings.base)
    return _excess_entropy(symbols, max_m, base=settings.base)


def _compute_permutation(values: npt.NDArray[np.generic], settings: _EntropySpec) -> float:
    """Return the entropy of the ordinal patterns of a series or of the windows of a field."""
    dx, dy = (settings.dimension, 1) if isinstance(settings.dimension, int) else settings.dimension
    array = _break_ties(values, settings.ties, settings.random_state)
    _, probabilities = ordpy.ordinal_distribution(array, dx=dx, dy=dy, taux=settings.delay, tauy=settings.delay)
    value = _measure_of(probabilities, settings.measure, settings.order)
    if settings.normalize:
        value /= _largest_measure(math.factorial(dx * dy), settings.measure, settings.order)
    return float(abs(value))


def _compute_regularity(array: Array, settings: _EntropySpec) -> float | Array:
    """Return the sample, approximate, multiscale, or spectral entropy of a real series."""
    antropy = _import('antropy')
    dimension = _get_length(settings)
    if settings.kind == 'sample':
        return _sample(array, dimension, settings.tolerance)
    if settings.kind == 'approximate':
        radius = _tolerance(array, settings.tolerance)
        return float(antropy.app_entropy(array, order=dimension, tolerance=radius))
    if settings.kind == 'multiscale':
        radius = _tolerance(array, settings.tolerance)
        entropies = []
        for scale in settings.scales:
            count = array.size // scale
            coarse = np.ascontiguousarray(array[: count * scale].reshape(count, scale).mean(axis=1))
            entropies.append(antropy.sample_entropy(coarse, order=dimension, tolerance=radius))
        return np.asarray(entropies, dtype=float)
    return float(antropy.spectral_entropy(array, sf=1 / settings.dt, method='fft', normalize=settings.normalize))


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
    """Return the entropy of the blocks of `m` consecutive symbols, or of square patches of side `m` of lattices."""
    array = np.asarray(symbols)
    if m == 0:
        return 0.0
    windows = np.lib.stride_tricks.sliding_window_view(array, (m,) * array.ndim)
    blocks = windows.reshape(-1, m**array.ndim)
    return _shannon(
        _probabilities([block.tobytes() for block in blocks]),
        base=base,
        probabilities=True,
    )


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
    return float(_import('antropy').sample_entropy(array, order=dimension, tolerance=radius)) + 0.0
