"""Test the `_real_time` module of `skcomplex.engine.runtime`."""

import time

import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.engine.runtime import RealTimeClock


def test_real_time_clock_run_waits_until_the_frame_is_due():
    """Test that running a real-time clock sleeps until a frame's simulated time is due in wall time."""
    clock = RealTimeClock(speed=10.0).prepare()
    clock.run(1.0)
    assert time.perf_counter() - clock.started_ >= 0.1


def test_real_time_clock_run_records_the_lag():
    """Test that a frame due in the past records its lag instead of waiting."""
    clock = RealTimeClock(speed=1.0).prepare()
    time.sleep(0.05)
    clock.run(0.0)
    assert clock.max_lag_ >= 0.05


@pytest.mark.parametrize('speed', [0, -1.0, 'fast', True])
def test_real_time_clock_check_rejects_a_speed_that_is_not_positive(speed):
    """Test that checking a real-time clock rejects a speed that is not a positive number."""
    with pytest.raises(ValidationError, match='speed'):
        RealTimeClock(speed=speed).check()
