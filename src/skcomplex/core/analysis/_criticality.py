"""Measure criticality: avalanches, crackling-noise scaling, branching ratios, and transitions of order parameters."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..base import AnalysisError
from ..compile import RelaxationRecord
from ._kinds import DataKind, analysis
from ._lazy import module
from ._statistics import PowerLawFit, fit_power_law

type Array = npt.NDArray[np.float64]
MIN_POINTS = 2
SHAPE_GRID = 50


def _records(source: object) -> list[RelaxationRecord]:
    """Return the relaxation records of a source, or nothing when it holds activity values."""
    if isinstance(source, np.ndarray) or not isinstance(source, Sequence):
        return []
    records = [item for item in source if isinstance(item, RelaxationRecord)]
    return records if records and len(records) == len(source) else []


@dataclass(frozen=True)
class Avalanches:
    """Avalanches of an activity series or of lattice relaxations.

    Args:
        sizes:
            The total activity of every avalanche.

        durations:
            The number of steps every avalanche lasts.

        shapes:
            The activity profile of every avalanche of an activity series, empty for relaxation records.
    """

    sizes: Array
    durations: Array
    shapes: list[Array]


@analysis(DataKind.SERIES)
def avalanches(source: npt.ArrayLike | Sequence[RelaxationRecord], threshold: float = 0.0) -> Avalanches:
    """Return the avalanches of an activity series, the runs above a threshold, or of the relaxations of a lattice.

    Args:
        source:
            The activity per step, such as spike counts or topplings, or the relaxation records of a trajectory.

        threshold:
            The activity an avalanche exceeds at every step, subtracted from its profile.

    Returns:
        avalanches:
            The sizes, the durations, and the profiles.

    Examples:
        >>> from skcomplex.core.analysis import avalanches
        >>> found = avalanches([0, 1, 2, 1, 0, 3, 0])
        >>> found.sizes.tolist(), found.durations.tolist()
        ([4.0, 3.0], [3.0, 1.0])
    """
    records = _records(source)
    if records:
        sizes = np.asarray([record.size for record in records], dtype=np.float64)
        return Avalanches(sizes, np.asarray([record.duration for record in records], dtype=np.float64), [])
    activity = np.asarray(source, dtype=float) - threshold
    active = np.concatenate([[False], activity > 0, [False]]).astype(int)
    starts, ends = np.flatnonzero(np.diff(active) == 1), np.flatnonzero(np.diff(active) == -1)
    shapes = [activity[start:end] for start, end in zip(starts, ends, strict=True)]
    return Avalanches(
        np.asarray([shape.sum() for shape in shapes], dtype=float),
        np.asarray([shape.size for shape in shapes], dtype=float),
        shapes,
    )


@dataclass(frozen=True)
class AvalancheLaws:
    """Power laws fitted to the sizes and the durations of avalanches.

    Args:
        size:
            The fit of the sizes, whose exponent is `tau_S`.

        duration:
            The fit of the durations, whose exponent is `tau_T`.
    """

    size: PowerLawFit
    duration: PowerLawFit


@analysis(DataKind.SERIES)
def fit_avalanches(
    found: Avalanches,
    *,
    size_range: tuple[float | None, float | None] = (None, None),
    duration_range: tuple[float | None, float | None] = (None, None),
    discrete: bool = True,
    bootstrap: int = 0,
    random_state: int | None = None,
) -> AvalancheLaws:
    """Return the power laws of the sizes and the durations of avalanches, compared with other tails.

    Args:
        found:
            The avalanches.

        size_range:
            The fixed lower and upper cutoffs of the sizes, `None` to choose the lower and to leave out the upper.

        duration_range:
            The fixed lower and upper cutoffs of the durations.

        discrete:
            Whether sizes and durations are integers.

        bootstrap:
            The number of bootstrap samples of the goodness of fit.

        random_state:
            The seed of the bootstrap.

    Returns:
        laws:
            The fits of the sizes and of the durations.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import Avalanches, fit_avalanches
        >>> rng = np.random.default_rng(0)
        >>> sizes = np.floor((1 - rng.random(5000)) ** (-1 / 0.5))
        >>> laws = fit_avalanches(Avalanches(sizes, sizes, []), size_range=(1, None), duration_range=(1, None))
        >>> round(laws.size.alpha, 0)
        1.0
    """

    def fit(values: Array, cutoffs: tuple[float | None, float | None]) -> PowerLawFit:
        """Return the power law of one observable."""
        low, high = cutoffs
        return fit_power_law(
            values,
            xmin=low,
            xmax=high,
            discrete=discrete,
            bootstrap=bootstrap,
            random_state=random_state,
        )

    return AvalancheLaws(fit(found.sizes, size_range), fit(found.durations, duration_range))


def _mean_shape(shapes: list[Array]) -> Array:
    """Return the mean profile of avalanches of one duration, scaled by the mean height, on a grid of 0 to 1."""
    profiles = np.asarray(shapes, dtype=float)
    grid = np.linspace(0, 1, SHAPE_GRID)
    steps = np.linspace(0, 1, profiles.shape[1])
    mean = profiles.mean(axis=0)
    return np.interp(grid, steps, mean / mean.mean())


@dataclass(frozen=True)
class CracklingNoise:
    """Check of the crackling-noise relation between avalanche exponents and collapse of mean shapes.

    Args:
        exponent:
            The exponent `gamma` of the mean size against the duration, `<S>(T) ~ T^gamma`.

        predicted:
            The exponent `(tau_T - 1) / (tau_S - 1)` from the fitted laws, or `nan` without laws.

        shape_spread:
            The mean over the inner 80 percent of the scaled time of the relative difference between the largest and
            the smallest mean shape of the chosen durations, scaled to unit mean height and unit duration, `nan`
            without shapes.

        shapes:
            The scaled mean shape of every chosen duration on a grid from 0 to 1.
    """

    exponent: float
    predicted: float
    shape_spread: float
    shapes: dict[int, Array]


@analysis(DataKind.SERIES)
def crackling_noise(
    found: Avalanches,
    laws: AvalancheLaws | None = None,
    durations: Sequence[int] = (),
    *,
    min_count: int = 10,
    min_duration: int = 1,
) -> CracklingNoise:
    """Return the exponent of the mean avalanche size against duration, its prediction from the laws, and shapes.

    Args:
        found:
            The avalanches.

        laws:
            The fitted size and duration laws, or `None` to leave out the prediction.

        durations:
            The durations whose mean shapes are compared.

        min_count:
            The fewest avalanches of a duration used in the fit of the mean size.

        min_duration:
            The shortest duration used in the fit of the mean size.

    Returns:
        crackling:
            The fitted and predicted exponents and the shape collapse.

    Raises:
        AnalysisError:
            Fewer than 2 durations have `min_count` avalanches.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import Avalanches, crackling_noise
        >>> durations = np.repeat(np.arange(1.0, 21.0), 20)
        >>> round(crackling_noise(Avalanches(durations**2, durations, [])).exponent, 3)
        2.0
    """
    distinct, counts = np.unique(found.durations, return_counts=True)
    kept = distinct[(counts >= min_count) & (distinct >= max(min_duration, 1))]
    if kept.size < MIN_POINTS:
        error_msg = f'Fewer than {MIN_POINTS} durations have {min_count} avalanches. Give more avalanches.'
        raise AnalysisError(error_msg)
    means = np.asarray([found.sizes[found.durations == duration].mean() for duration in kept])
    exponent = float(np.polyfit(np.log(kept), np.log(means), 1)[0])
    predicted = float('nan') if laws is None else (laws.duration.alpha - 1) / (laws.size.alpha - 1)
    shapes = {
        int(duration): _mean_shape([shape for shape in found.shapes if shape.size == duration])
        for duration in durations
        if any(shape.size == duration for shape in found.shapes)
    }
    spread = float('nan')
    if len(shapes) >= MIN_POINTS:
        stacked = np.asarray(list(shapes.values()))
        inner = slice(SHAPE_GRID // 10, SHAPE_GRID - SHAPE_GRID // 10)
        spread = float(np.mean((stacked.max(axis=0) - stacked.min(axis=0))[inner] / stacked.mean(axis=0)[inner]))
    return CracklingNoise(exponent, float(predicted), spread, shapes)


@analysis(DataKind.SERIES)
def branching_ratio(activity: npt.ArrayLike, max_step: int = 40, *, dt: float = 1.0) -> float:
    """Return the branching ratio of an activity series by the multistep regression estimator of module('mrestimator').

    The estimator fits `r_k ∝ m^k` to the regression slopes `r_k` of the activity `k` steps apart, which stays
    unbiased when only a fraction of the units is observed.

    Args:
        activity:
            The activity per step, or one row per trial.

        max_step:
            The largest step `k` of the regression.

        dt:
            The duration of a step.

    Returns:
        ratio:
            The branching ratio `m`, 1 at criticality.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import branching_ratio
        >>> rng = np.random.default_rng(0)
        >>> activity = np.zeros(20_000)
        >>> activity[0] = 100
        >>> for t in range(1, activity.size):
        ...     activity[t] = rng.poisson(0.9 * activity[t - 1] + 10)
        >>> round(branching_ratio(activity), 1)
        0.9
    """
    data = np.asarray(activity, dtype=float)
    data = data.reshape(1, -1) if data.ndim == 1 else data
    coefficients = module('mrestimator').coefficients(
        data,
        method='ts',
        steps=(1, max_step),
        dt=dt,
        dtunit='step',
        numboot=0,
    )
    result = module('mrestimator').fit(coefficients, fitfunc=module('mrestimator').f_exponential_offset, numboot=0)
    return float(result.mre)


@dataclass(frozen=True)
class Sweep:
    """Order parameter, susceptibility, and Binder cumulant over a sweep of a control parameter.

    Args:
        parameters:
            The control parameter values.

        order:
            The mean absolute order parameter at every value.

        susceptibility:
            The fluctuation `size (<m²> - <|m|>²)`, divided by the temperature when one is given.

        binder:
            The Binder cumulant at every value.

        peak:
            The control parameter value of the largest susceptibility, refined by a parabola through its neighbours.
    """

    parameters: Array
    order: Array
    susceptibility: Array
    binder: Array
    peak: float


def _refined_peak(grid: Array, values: Array) -> float:
    """Return the abscissa of the vertex of the parabola through the largest value and its neighbours."""
    index = int(np.argmax(values))
    if index in (0, values.size - 1):
        return float(grid[index])
    coefficients = np.polyfit(grid[index - 1 : index + 2], values[index - 1 : index + 2], 2)
    if coefficients[0] >= 0:
        return float(grid[index])
    return float(-coefficients[1] / (2 * coefficients[0]))


@analysis(DataKind.ENSEMBLE)
def order_parameter_sweep(
    parameters: npt.ArrayLike,
    samples: Sequence[npt.ArrayLike],
    size: float,
    *,
    temperatures: npt.ArrayLike | None = None,
) -> Sweep:
    """Return the mean order parameter, the susceptibility from its fluctuations, and the Binder cumulant of a sweep.

    Args:
        parameters:
            The control parameter values, such as temperatures or couplings.

        samples:
            The samples of the order parameter per unit size at every value, over time or over disorder.

        size:
            The system size `N` that scales the susceptibility.

        temperatures:
            The temperature at every value that divides the susceptibility, or `None` to leave it undivided.

    Returns:
        sweep:
            The order parameter, the susceptibility, the cumulant, and the peak of the susceptibility.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import order_parameter_sweep
        >>> rng = np.random.default_rng(0)
        >>> samples = [rng.normal(0, width, 1000) for width in (0.1, 0.5, 0.2)]
        >>> order_parameter_sweep([1.0, 2.0, 3.0], samples, 100).peak > 1.5
        True
    """
    grid = np.asarray(parameters, dtype=float)
    arrays = [np.abs(np.asarray(sample, dtype=float)) for sample in samples]
    order = np.asarray([array.mean() for array in arrays])
    susceptibility = size * np.asarray([np.mean(array**2) - array.mean() ** 2 for array in arrays])
    if temperatures is not None:
        susceptibility = susceptibility / np.asarray(temperatures, dtype=float)
    binder = np.asarray([1 - np.mean(array**4) / (3 * np.mean(array**2) ** 2) for array in arrays])
    return Sweep(grid, order, susceptibility, binder, _refined_peak(grid, susceptibility))


@analysis(DataKind.FIELD)
def replica_overlap(first: npt.ArrayLike, second: npt.ArrayLike) -> Array | float:
    """Return the overlap `q = <s_i t_i>` of two replicas of a configuration, per frame for stacks of frames.

    Args:
        first:
            The spins or states of the first replica, one configuration or one per frame.

        second:
            The second replica, shaped like the first.

    Returns:
        overlap:
            The overlap, ±1 for identical or reversed configurations and near 0 for independent ones.

    Examples:
        >>> from skcomplex.core.analysis import replica_overlap
        >>> replica_overlap([1, -1, 1, 1], [1, -1, -1, 1])
        0.5
    """
    product = np.asarray(first, dtype=float) * np.asarray(second, dtype=float)
    if product.ndim == 1:
        return float(product.mean())
    overlap = product.reshape(product.shape[0], -1).mean(axis=1)
    return overlap


@analysis(DataKind.ENSEMBLE)
def fit_order_exponent(
    parameters: npt.ArrayLike,
    order: npt.ArrayLike,
    critical: float,
    *,
    above: bool = False,
) -> float:
    """Return the exponent `beta` of the order parameter `m ~ |p - p_c|^beta` on the ordered side of a transition.

    Args:
        parameters:
            The control parameter values.

        order:
            The order parameter at every value.

        critical:
            The critical value.

        above:
            Whether the ordered side lies above the critical value, as for couplings, rather than below it, as for
            temperatures.

    Returns:
        beta:
            The exponent, 1/2 in mean field.

    Raises:
        AnalysisError:
            Fewer than 2 values with a positive order lie on the ordered side.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import fit_order_exponent
        >>> temperatures = np.linspace(0.9, 0.99, 10)
        >>> round(fit_order_exponent(temperatures, np.sqrt(1 - temperatures), 1.0), 6)
        0.5
    """
    grid = np.asarray(parameters, dtype=float)
    values = np.asarray(order, dtype=float)
    distance = grid - critical if above else critical - grid
    ordered = (distance > 0) & (values > 0)
    if ordered.sum() < MIN_POINTS:
        error_msg = f'Fewer than {MIN_POINTS} values with a positive order lie on the ordered side of {critical}.'
        raise AnalysisError(error_msg)
    return float(np.polyfit(np.log(distance[ordered]), np.log(values[ordered]), 1)[0])
