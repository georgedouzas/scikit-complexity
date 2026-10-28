"""Share the input checks of the analysis functions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import numpy as np
import numpy.typing as npt

from ..core.errors import ValidationError


def real_array(values: npt.ArrayLike, argument: str) -> npt.NDArray[np.float64]:
    """Return values as a float array after rejecting complex values, whose imaginary part a cast would drop.

    Args:
        values:
            The values an analysis function receives.

        argument:
            The name of the argument that holds the values, used in the error message.

    Returns:
        array:
            The values as floats.

    Raises:
        ValidationError:
            The values are complex.
    """
    array = np.asarray(values)
    if np.iscomplexobj(array):
        error_msg = (
            f'The {argument} are complex, and the analysis takes real values. Pass one real view of them: the real '
            'part `values.real`, the imaginary part `values.imag`, the modulus `np.abs(values)`, or the argument '
            '`np.angle(values)`.'
        )
        raise ValidationError(error_msg)
    return np.asarray(array, dtype=float)


def sample_columns(data: npt.ArrayLike) -> npt.NDArray[Any]:
    """Return data as one row per sample and one column per variable, a series as a single column.

    Args:
        data:
            A series, or one row per sample with the variables along the remaining axes.

    Returns:
        columns:
            The data as a two-dimensional array.
    """
    array = np.asarray(data)
    return array.reshape(-1, 1) if array.ndim == 1 else array.reshape(array.shape[0], -1)


def is_changed(value: object, default: object) -> bool:
    """Return whether a setting differs from its default, comparing arrays and sequences by their items.

    Args:
        value:
            The value of the setting.

        default:
            The default of the setting.

    Returns:
        changed:
            Whether the value differs from the default.
    """
    if value is None or default is None:
        return value is not default
    return not np.array_equal(np.asarray(value, dtype=object), np.asarray(default, dtype=object))
