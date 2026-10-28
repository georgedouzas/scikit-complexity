"""Measure response to a field and the fluctuation-dissipation ratio of aging systems."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt

from ._kinds import DataKind, analysis

type Array = npt.NDArray[np.float64]


@analysis(DataKind.ENSEMBLE)
def response(perturbed: npt.ArrayLike, unperturbed: npt.ArrayLike, field: float) -> Array:
    """Return the integrated response `chi(t) = (<x(t)>_h - <x(t)>_0) / h` of ensembles with and without a field.

    Args:
        perturbed:
            One row per run and one column per frame, with the field switched on.

        unperturbed:
            The same without the field.

        field:
            The strength of the field.

    Returns:
        response:
            The integrated response at every frame.

    Examples:
        >>> from skcomplex.analysis import response
        >>> response([[1.0, 2.0]], [[1.0, 1.0]], 0.5).tolist()
        [0.0, 2.0]
    """
    first, second = np.asarray(perturbed, dtype=float), np.asarray(unperturbed, dtype=float)
    return (first.mean(axis=0) - second.mean(axis=0)) / field


@analysis(DataKind.ENSEMBLE)
def fluctuation_dissipation_ratio(
    correlation: npt.ArrayLike,
    susceptibility: npt.ArrayLike,
    temperature: float,
) -> float:
    """Return the fluctuation-dissipation ratio, the slope of `T chi` against `C(t_w, t_w) - C(t, t_w)`.

    Args:
        correlation:
            The two-time correlation over lags from 0.

        susceptibility:
            The integrated response over the same lags.

        temperature:
            The temperature.

    Returns:
        ratio:
            1 in equilibrium, below 1 in an aging system.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import fluctuation_dissipation_ratio
        >>> lags = np.linspace(0, 3, 30)
        >>> round(fluctuation_dissipation_ratio(np.exp(-lags), 1 - np.exp(-lags), 1.0), 6)
        1.0
    """
    values = np.asarray(correlation, dtype=float)
    decay = values[0] - values
    scaled = temperature * np.asarray(susceptibility, dtype=float)
    return float(np.sum(decay * scaled) / np.sum(decay**2))
