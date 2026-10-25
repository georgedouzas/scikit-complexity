"""Test the `_plantations` module of `skcomplex.models.economics.resources`."""

import pytest

from skcomplex.models.economics.resources import (
    cocoa,
)

from ...conftest import _peak, _run_monitored


@pytest.mark.slow
def test_cocoa_booms_and_busts_without_a_subsidy():
    """Test that the cocoa area booms, then collapses under the insects, unless a full subsidy keeps the shade."""
    bust = _run_monitored(cocoa(), 0.25, 50)
    area = bust.values('cocoa.area')
    assert area.max() > 10 * area[0]
    assert 5 < _peak(bust, 'cocoa.area') < 25
    assert area[-1] < 0.05 * area.max()
    assert bust.values('cocoa.insects')[-1] > 90
    sustained = _run_monitored(cocoa(subsidy=1.0), 0.25, 50).values('cocoa.area')
    assert sustained[-1] > 0.9 * sustained.max()
