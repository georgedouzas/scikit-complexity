"""Test the `_schedule_spec` module of `skcomplex.core.components.spaces`."""

from skcomplex.core.components.spaces import ScheduleSpec


def test_schedule_spec_has_nothing_to_check():
    """Test that the base schedule spec checks into itself."""
    schedule = ScheduleSpec()
    schedule._check()
