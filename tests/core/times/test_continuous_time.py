"""Test the `_continuous_time` module of `skcomplex.core.times`."""

from skcomplex.core.times import ContinuousTime


def test_continuous_time_has_no_schedule_and_spans_the_run():
    """Test that continuous time prepares no schedule and frames the start and the end of a run."""
    time = ContinuousTime().check().prepare()
    assert time.schedule_ is None
    assert time._get_frames(3.0).tolist() == [0.0, 3.0]
