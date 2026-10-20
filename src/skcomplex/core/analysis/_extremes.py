"""Describe extremes and limit laws: extreme-value fits, records, stable laws, and goodness of fit to named laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from ..base import AnalysisError
from ._kinds import DataKind, analysis

type Array = npt.NDArray[np.float64]
LAWS = {
    'exponential': stats.expon,
    'normal': stats.norm,
    'lognormal': stats.lognorm,
    'gamma': stats.gamma,
    'maxwell': stats.maxwell,
    'rayleigh': stats.rayleigh,
    'uniform': stats.uniform,
}


@dataclass(frozen=True)
class ExtremeValueFit:
    """Generalized extreme-value law fitted to block maxima.

    Args:
        shape:
            The shape `xi`, 0 for the Gumbel class, positive for the Fréchet class of heavy tails, and negative for
            the Weibull class of bounded tails.

        location:
            The location.

        scale:
            The scale.

        maxima:
            The maximum of every block.
    """

    shape: float
    location: float
    scale: float
    maxima: Array


@analysis(DataKind.SERIES)
def fit_extreme_values(values: npt.ArrayLike, block: int) -> ExtremeValueFit:
    """Return the generalized extreme-value law fitted by maximum likelihood to the maxima of blocks of samples.

    Args:
        values:
            The samples.

        block:
            The number of samples per block.

    Returns:
        fit:
            The shape, the location, the scale, and the maxima.

    Raises:
        AnalysisError:
            The samples fill fewer than 10 blocks.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import fit_extreme_values
        >>> samples = np.random.default_rng(0).exponential(size=200_000)
        >>> round(fit_extreme_values(samples, 100).shape, 1)
        0.0
    """
    array = np.asarray(values, dtype=float)
    count = array.size // block
    if count < 10:  # noqa: PLR2004
        error_msg = f'{array.size} samples fill {count} blocks of {block}. A fit needs at least 10 blocks.'
        raise AnalysisError(error_msg)
    maxima = array[: count * block].reshape(count, block).max(axis=1)
    shape, location, scale = stats.genextreme.fit(maxima)
    return ExtremeValueFit(float(-shape), float(location), float(scale), maxima)


@analysis(DataKind.ENSEMBLE)
def record_counts(sequences: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Return the number of records of every sequence, the values larger than every value before them.

    Args:
        sequences:
            One sequence, or one row per sequence.

    Returns:
        counts:
            The number of records of every sequence, the first value included, about the harmonic number of the
            length for independent values.

    Examples:
        >>> from skcomplex.core.analysis import record_counts
        >>> record_counts([[1, 3, 2, 5], [4, 3, 2, 1]]).tolist()
        [3, 1]
    """
    array = np.asarray(sequences, dtype=float)
    array = array.reshape(1, -1) if array.ndim == 1 else array
    previous = np.maximum.accumulate(array, axis=1)
    records = np.concatenate([np.ones((array.shape[0], 1), dtype=bool), array[:, 1:] > previous[:, :-1]], axis=1)
    return records.sum(axis=1).astype(np.int64)


def _starting_point(_: object, start: Array, *__: object, **___: object) -> Array:
    """Return SciPy's starting point of a fit, McCulloch's quantile estimate for stable laws, unchanged."""
    return start


@dataclass(frozen=True)
class StableFit:
    """Lévy stable law fitted to samples.

    Args:
        alpha:
            The stability index, 2 for the normal law.

        beta:
            The skewness.

        location:
            The location.

        scale:
            The scale.
    """

    alpha: float
    beta: float
    location: float
    scale: float


@analysis(DataKind.SERIES)
def fit_stable(values: npt.ArrayLike) -> StableFit:
    """Return the Lévy stable law fitted to samples by McCulloch's quantile method, in SciPy's `S1` parameterization.

    Args:
        values:
            The samples.

    Returns:
        fit:
            The index, the skewness, the location, and the scale.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import fit_stable
        >>> round(fit_stable(np.random.default_rng(0).standard_cauchy(500)).alpha, 0)
        1.0
    """
    alpha, beta, location, scale = stats.levy_stable.fit(np.asarray(values, dtype=float), optimizer=_starting_point)
    return StableFit(float(alpha), float(beta), float(location), float(scale))


@dataclass(frozen=True)
class GoodnessOfFit:
    """Test of samples against a named law with fitted parameters.

    Args:
        law:
            The name of the law.

        parameters:
            The fitted and fixed parameters of the law.

        statistic:
            The Kolmogorov-Smirnov statistic.

        p_value:
            The Monte Carlo p-value, which accounts for the fitted parameters. Values below the significance level
            reject the law.
    """

    law: str
    parameters: dict[str, float]
    statistic: float
    p_value: float


@analysis(DataKind.SERIES)
def compare_with_law(
    values: npt.ArrayLike,
    law: str = 'exponential',
    *,
    fixed: Mapping[str, float] | None = None,
    simulations: int = 199,
    random_state: int | None = 0,
) -> GoodnessOfFit:
    """Test samples against a named law whose free parameters are fitted, with a Monte Carlo Kolmogorov-Smirnov test.

    Args:
        values:
            The samples.

        law:
            `'exponential'`, `'normal'`, `'lognormal'`, `'gamma'`, `'maxwell'`, `'rayleigh'`, or `'uniform'`.

        fixed:
            The parameters held fixed by SciPy name, such as `{'loc': 0}`, or `None` to fit all.

        simulations:
            The number of Monte Carlo samples of the null distribution.

        random_state:
            The seed of the Monte Carlo samples.

    Returns:
        fit:
            The law, its parameters, the statistic, and the p-value.

    Raises:
        AnalysisError:
            The law is unknown.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import compare_with_law
        >>> fit = compare_with_law(np.random.default_rng(0).exponential(2.0, 1000), fixed={'loc': 0})
        >>> round(fit.parameters['scale'], 0), fit.p_value > 0.05
        (2.0, True)
    """
    if law not in LAWS:
        error_msg = f'Unknown law {law!r}. Use one of {", ".join(LAWS)}.'
        raise AnalysisError(error_msg)
    result = stats.goodness_of_fit(
        LAWS[law],
        np.asarray(values, dtype=float),
        known_params=dict(fixed or {}),
        statistic='ks',
        n_mc_samples=simulations,
        rng=np.random.default_rng(random_state),
    )
    parameters = {name: float(value) for name, value in result.fit_result.params._asdict().items()}
    return GoodnessOfFit(law, parameters, float(result.statistic), float(result.pvalue))
