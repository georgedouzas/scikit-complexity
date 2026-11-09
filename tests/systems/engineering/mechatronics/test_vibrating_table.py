"""Test the `_vibrating_table` module of `skcomplex.systems.engineering.mechatronics`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.engineering.mechatronics import VibratingTableSystem


def test_vibrating_table_settles_on_a_steady_vibration_close_to_the_linear_model():
    """Test that the table vibrates steadily at the supply frequency, with nearly the amplitude of the linear model."""
    swings = []
    for linear in (False, True):
        trajectory = simulate(VibratingTableSystem(linear=linear))
        times, x = trajectory.times, np.ravel(trajectory.get_values('shaker.x'))
        late = times > 20
        swing = [np.ptp(x[(times > start) & (times < start + 4)]) for start in (20, 28, 36)]
        assert np.ptp(swing) < 1e-3 * swing[0]
        rising = times[late][1:][(x[late][:-1] < 0) & (x[late][1:] >= 0)]
        assert np.mean(np.diff(rising)) == pytest.approx(2 * np.pi / 1.5915, rel=0.01)
        swings.append(swing[0])
    assert swings[0] == pytest.approx(swings[1], rel=0.05)
