"""Measure the regularity and complexity of series: sample, multiscale, and spectral entropies and their planes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import ordpy

from ..core.errors import AnalysisError
from ._entropy import _break_ties, _shannon
from ._information import _kl_divergence
from ._kinds import DataKind, analysis

type Array = npt.NDArray[np.float64]


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
        >>> from skcomplex.analysis import complexity_entropy
        >>> plane = complexity_entropy(np.random.default_rng(0).standard_normal(50_000), 4)
        >>> round(plane.entropy, 2), round(plane.complexity, 2)
        (1.0, 0.0)
    """
    dx, dy = (order, 1) if isinstance(order, int) else order
    array = _break_ties(values, ties, random_state)
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
        >>> from skcomplex.analysis import ensemble_entropy
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
        np.asarray([_shannon(histogram, probabilities=True) for histogram in histograms]),
        np.asarray([_kl_divergence(histogram, law) for histogram in histograms]),
        float((np.count_nonzero(law) - 1) / (2 * array.shape[0] * np.log(2))),
    )
