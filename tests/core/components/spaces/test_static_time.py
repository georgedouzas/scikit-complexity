"""Test the `_static_time` module of `skcomplex.core.components.spaces`."""

from skcomplex.core.components.spaces._static_time import _StaticTime


def test_static_time_has_no_schedule():
    """Test that static time prepares no schedule."""
    time = _StaticTime().prepare()
    assert time.schedule_ is None
