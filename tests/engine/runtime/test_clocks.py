"""Test the `_clocks` module of `skcomplex.engine.runtime`."""

import time

from skcomplex.engine.runtime import AsFastAsPossibleClock, RealTimeClock


def test_clock_fit_marks_the_start():
    """Test that preparing a clock marks the wall time of the start and no lag."""
    before = time.perf_counter()
    clock = RealTimeClock(speed=10.0).prepare()
    assert before <= clock.started_ <= time.perf_counter()
    assert clock.max_lag_ == 0.0


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


def test_as_fast_as_possible_clock_never_waits():
    """Test that the default clock returns at once and has only the estimator verbs."""
    clock = AsFastAsPossibleClock().prepare()
    start = time.perf_counter()
    clock.run(1e6)
    assert time.perf_counter() - start < 0.1
    assert not hasattr(clock, 'start')
    assert not hasattr(clock, 'wait')
