"""Share the helpers and fixtures of the tests below this directory."""

import numpy as np

from skcomplex.engine.runtime import Trajectory


def _stationary_polarization(trajectory: Trajectory, start: int) -> float:
    """Return the polarization of a trajectory averaged over its frames from `start` on."""
    return float(np.mean(np.abs(np.asarray(trajectory['polarization'])[start:])))
