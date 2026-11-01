"""Test the `_base` module of `skcomplex.core.times`."""

from types import SimpleNamespace

from skcomplex.core.times import BaseTime


class TickTime(BaseTime):
    """Time that takes the frames of the base."""

    def __init__(self, *, label='time'):
        super().__init__(label)

    def _prepare(self, context=None):
        """Set no schedule."""
        self.schedule_ = None


def test_base_time_runs_into_no_laws_and_frames_its_ends():
    """Test that the base time contributes no laws and frames the start and the end of a run."""
    time = TickTime().check().prepare()
    assert time.schedule_ is None
    assert not hasattr(time.run(SimpleNamespace()), 'laws_')
    assert time._get_frames(2.0).tolist() == [0.0, 2.0]
