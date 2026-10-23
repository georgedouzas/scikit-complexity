"""Share the helper of the statistical mechanics model tests."""

import numpy as np

from skcomplex.core.runtime import Trajectory


def magnetization(trajectory: Trajectory) -> float:
    """Return the mean absolute magnetisation of the recorded frames."""
    return float(np.abs(trajectory['spins.spin'].mean(axis=1)).mean())
