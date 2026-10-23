"""Test the `_diversity` module of `skcomplex.models.economics.innovation`."""

import pytest

from skcomplex.models.economics.innovation import (
    creative_phase_transition,
)

from ..conftest import _run


@pytest.mark.parametrize(('r', 'low'), [(2.0, True), (3.0, False)])
def test_creative_phase_transition_jumps_above_the_critical_rule_density(r, low):
    """Test that the final diversity stays low below the critical rule density and jumps to almost full above it."""
    diversity = _run(creative_phase_transition(0.1, r=r), 100, dt=1)['system.x'][-1]
    assert (diversity < 0.2) if low else (diversity > 0.9)
