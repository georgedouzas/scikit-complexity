"""Share the input checks of the analysis functions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt

from ..base import ValidationError


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
