"""Test the `_as_fast_as_possible` module of `skcomplex.core.engine.runtime`."""

import time

from skcomplex.core.engine.runtime import AsFastAsPossibleClock


def test_as_fast_as_possible_clock_never_waits():
    """Test that the default clock returns at once and has only the estimator verbs."""
    clock = AsFastAsPossibleClock().prepare()
    start = time.perf_counter()
    clock.run(1e6)
    assert time.perf_counter() - start < 0.1
    assert not hasattr(clock, 'start')
    assert not hasattr(clock, 'wait')
