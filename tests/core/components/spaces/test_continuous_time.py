"""Test the `_continuous_time` module of `skcomplex.core.components.spaces`."""

from skcomplex.core.components.spaces._continuous_time import _ContinuousTime


def test_continuous_time_has_no_schedule():
    """Test that continuous time prepares no schedule."""
    time = _ContinuousTime().prepare()
    assert time.schedule_ is None
