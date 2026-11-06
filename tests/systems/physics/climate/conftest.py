"""Share the helpers of the climate system tests."""

import numpy as np
import numpy.typing as npt


def _sign_changes(values: npt.ArrayLike) -> int:
    """Return the number of times a series changes sign."""
    return int((np.diff(np.sign(values)) != 0).sum())
