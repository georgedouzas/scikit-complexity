"""Share the helper of the statistical mechanics system tests."""

import numpy as np

from skcomplex.core.engine.runtime import Trajectory


def magnetization(trajectory: Trajectory) -> float:
    """Return the mean absolute magnetisation of the recorded frames."""
    return float(np.abs(trajectory.get_values('spins.spin').mean(axis=1)).mean())
