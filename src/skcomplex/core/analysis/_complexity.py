"""Measure the regularity and complexity of series: sample, multiscale, and spectral entropies and their planes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import ordpy

from ..base import AnalysisError
from ._information import break_ties, kl_divergence, shannon_entropy
from ._kinds import DataKind, analysis
from ._lazy import module

type Array = npt.NDArray[np.float64]


def _tolerance(array: Array, tolerance: float) -> float:
    """Return the tolerance as a fraction of the standard deviation of a series."""
    return float(tolerance * array.std())


@analysis(DataKind.SERIES)
def sample_entropy(values: npt.ArrayLike, dimension: int = 2, tolerance: float = 0.2) -> float:
    """Return the sample entropy of a series, the negative log probability that close patterns stay close.

    Args:
        values:
            The series.

        dimension:
            The length `m` of the compared patterns.

        tolerance:
            The largest Chebyshev distance of close patterns, as a fraction of the standard deviation.

    Returns:
        entropy:
            `-ln(A / B)` in nats, with `A` and `B` the numbers of close pairs of length `m + 1` and `m`, 0 for a
            regular series and about 2.2 for white noise.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import sample_entropy
        >>> sample_entropy([0.2, 0.8] * 500)
        0.0
    """
    array = np.ascontiguousarray(values, dtype=float)
    return float(module('antropy').sample_entropy(array, order=dimension, tolerance=_tolerance(array, tolerance))) + 0.0


@analysis(DataKind.SERIES)
def approximate_entropy(values: npt.ArrayLike, dimension: int = 2, tolerance: float = 0.2) -> float:
    """Return the approximate entropy of a series, Pincus's regularity statistic with self-matches.

    Args:
        values:
            The series.

        dimension:
            The length of the compared patterns.

        tolerance:
            The largest Chebyshev distance of close patterns, as a fraction of the standard deviation.

    Returns:
        entropy:
            The approximate entropy in nats.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import approximate_entropy
        >>> bool(approximate_entropy(np.random.default_rng(0).standard_normal(1000)) > 1)
        True
    """
    array = np.ascontiguousarray(values, dtype=float)
    return float(module('antropy').app_entropy(array, order=dimension, tolerance=_tolerance(array, tolerance)))


@dataclass(frozen=True)
class Multiscale:
    """Sample entropy of a series coarse-grained over scales.

    Args:
        scales:
            The coarse-graining scales.

        entropies:
            The sample entropy at every scale.
    """

    scales: npt.NDArray[np.int64]
    entropies: Array


@analysis(DataKind.SERIES)
def multiscale_entropy(
    values: npt.ArrayLike,
    scales: Sequence[int] = tuple(range(1, 11)),
    dimension: int = 2,
    tolerance: float = 0.2,
) -> Multiscale:
    """Return Costa's multiscale entropy, the sample entropy of the means of non-overlapping windows of every scale.

    The tolerance stays a fraction of the standard deviation of the original series at every scale, so white noise
    loses entropy with the scale while long-range correlated noise keeps it.

    Args:
        values:
            The series.

        scales:
            The window lengths.

        dimension:
            The length of the compared patterns.

        tolerance:
            The largest distance of close patterns, as a fraction of the standard deviation of the series.

    Returns:
        multiscale:
            The scales and the sample entropy at every scale.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import multiscale_entropy
        >>> entropies = multiscale_entropy(np.random.default_rng(0).standard_normal(5000), (1, 5)).entropies
        >>> bool(entropies[1] < entropies[0])
        True
    """
    array = np.ascontiguousarray(values, dtype=float)
    radius = _tolerance(array, tolerance)
    entropies = []
    for scale in scales:
        count = array.size // scale
        coarse = array[: count * scale].reshape(count, scale).mean(axis=1)
        entropies.append(
            module('antropy').sample_entropy(np.ascontiguousarray(coarse), order=dimension, tolerance=radius),
        )
    return Multiscale(np.asarray(scales, dtype=np.int64), np.asarray(entropies, dtype=float))


@analysis(DataKind.SERIES)
def spectral_entropy(values: npt.ArrayLike, dt: float = 1.0, *, normalize: bool = True) -> float:
    """Return the spectral entropy of a series, the Shannon entropy of its normalized periodogram.

    Args:
        values:
            The series.

        dt:
            The time between samples.

        normalize:
            Whether to divide by the log of the number of frequencies, so the result lies between 0 and 1.

    Returns:
        entropy:
            The spectral entropy in bits, near 0 for a pure tone and near 1 normalized for white noise.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import spectral_entropy
        >>> bool(spectral_entropy(np.sin(2 * np.pi * 64 * np.arange(4096) / 4096)) < 0.1)
        True
    """
    array = np.ascontiguousarray(values, dtype=float)
    return float(
        module('antropy').spectral_entropy(array, sf=1 / dt, method='fft', normalize=normalize),
    )


@dataclass(frozen=True)
class ComplexityEntropy:
    """Position of data on the complexity-entropy plane of ordinal patterns, with the bounds of the plane.

    Args:
        entropy:
            The normalized permutation entropy.

        complexity:
            The statistical complexity, the entropy times the Jensen-Shannon divergence from uniform patterns.

        minimum:
            The curve of least complexity, one row of entropy and complexity per point.

        maximum:
            The curve of largest complexity, one row of entropy and complexity per point.
    """

    entropy: float
    complexity: float
    minimum: Array
    maximum: Array


@analysis(DataKind.SERIES)
def complexity_entropy(
    values: npt.ArrayLike,
    order: int | tuple[int, int] = 6,
    *,
    delay: int = 1,
    ties: str = 'order',
    random_state: int | None = None,
) -> ComplexityEntropy:
    """Return the place of a series or a two-dimensional field on Rosso's complexity-entropy plane.

    Noise has entropy near 1 and complexity near 0, regular series both near 0, and chaos lies in between with high
    complexity.

    Args:
        values:
            The series, or a two-dimensional field when `order` is a window.

        order:
            The length of the patterns of a series, or the window `(columns, rows)` of the patterns of a field.

        delay:
            The number of steps between the values of a pattern.

        ties:
            `'order'` to rank equal values by position, or `'random'` to rank them at random.

        random_state:
            The seed of the random ranks of equal values.

    Returns:
        plane:
            The entropy, the complexity, and the curves of least and largest complexity.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import complexity_entropy
        >>> plane = complexity_entropy(np.random.default_rng(0).standard_normal(50_000), 4)
        >>> round(plane.entropy, 2), round(plane.complexity, 2)
        (1.0, 0.0)
    """
    dx, dy = (order, 1) if isinstance(order, int) else order
    array = break_ties(values, ties, random_state)
    entropy, complexity = ordpy.complexity_entropy(array, dx=dx, dy=dy, taux=delay, tauy=delay)
    minimum = np.asarray(ordpy.minimum_complexity_entropy(dx=dx, dy=dy), dtype=float)
    maximum = np.asarray(ordpy.maximum_complexity_entropy(dx=dx, dy=dy), dtype=float)
    return ComplexityEntropy(float(entropy), float(complexity), minimum, maximum)


@dataclass(frozen=True)
class EnsembleEntropy:
    """Entropy of the distribution of an ensemble over outcomes at every frame, and its divergence from a law.

    Args:
        entropy:
            The Shannon entropy in bits of the histogram of every frame.

        divergence:
            The Kullback-Leibler divergence in bits of the histogram of every frame from the reference law.

        error:
            The expected divergence of a sample of the reference law of the ensemble's size, `(k - 1) / (2 n ln 2)`
            for `k` outcomes and `n` runs, below which a divergence is sampling noise.
    """

    entropy: Array
    divergence: Array
    error: float


@analysis(DataKind.ENSEMBLE)
def ensemble_entropy(samples: npt.ArrayLike, reference: npt.ArrayLike) -> EnsembleEntropy:
    """Return the entropy of an ensemble at every frame and its divergence from a reference law, as in the H-theorem.

    Args:
        samples:
            One row per run and one column per frame of integer outcomes from 0 to the number of outcomes minus 1.

        reference:
            The probability of every outcome under the reference law, such as the stationary law.

    Returns:
        entropy:
            The entropy, the divergence, and the sampling error of the divergence.

    Raises:
        AnalysisError:
            An outcome lies outside the reference law.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import ensemble_entropy
        >>> runs = np.random.default_rng(0).integers(0, 2, (10_000, 3))
        >>> bool(ensemble_entropy(runs, [0.5, 0.5]).divergence.max() < 0.001)
        True
    """
    array = np.asarray(samples, dtype=np.int64)
    law = np.asarray(reference, dtype=float)
    if array.min() < 0 or array.max() >= law.size:
        error_msg = f'Outcomes must lie between 0 and {law.size - 1}, the outcomes of the reference law.'
        raise AnalysisError(error_msg)
    histograms = [np.bincount(frame, minlength=law.size) / frame.size for frame in array.T]
    return EnsembleEntropy(
        np.asarray([shannon_entropy(histogram, probabilities=True) for histogram in histograms]),
        np.asarray([kl_divergence(histogram, law) for histogram in histograms]),
        float((np.count_nonzero(law) - 1) / (2 * array.shape[0] * np.log(2))),
    )
