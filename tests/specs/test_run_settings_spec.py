"""Test the run settings spec."""

import pytest

from skcomplex.errors import ValidationError
from skcomplex.specs import RunSettingsSpec


def test_check_accepts_positive_settings():
    """Test that positive settings pass, and that the start may be negative."""
    settings = RunSettingsSpec(start=-5.0, until=10.0, dt=0.25, save_every=1.0)
    assert settings.until == 10.0


@pytest.mark.parametrize('name', ['until', 'dt', 'save_every'])
def test_check_rejects_a_setting_that_is_not_positive(name):
    """Test that a run length, a time step, or a saving interval that is not positive raises an error."""
    values = {'start': 0.0, 'until': 10.0, 'dt': 0.25, 'save_every': 1.0, name: 0.0}
    with pytest.raises(ValidationError, match=name):
        RunSettingsSpec(**values).prepare()


def test_settings_are_frozen_values():
    """Test that settings compare by their fields."""
    assert RunSettingsSpec(0.0, 1.0, 0.5, 0.5) == RunSettingsSpec(0.0, 1.0, 0.5, 0.5)
