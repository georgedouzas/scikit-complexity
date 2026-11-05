"""Test the `_static_time` module of `skcomplex.core.components.spaces`."""

from skcomplex.core.components.spaces import StaticTime


def test_static_time_has_no_schedule_and_solves_once():
    """Test that static time prepares no schedule and has the one frame of the solution."""
    time = StaticTime().prepare()
    assert time.schedule_ is None
    assert time._get_frames(3.0).tolist() == [0.0]
