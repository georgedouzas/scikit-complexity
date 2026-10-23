"""Test the `_succession` module of `skcomplex.models.biology.ecology`."""

from skcomplex.models.biology.ecology import (
    mangrove_succession,
)

from ..conftest import _run_until


def test_mangrove_pioneers_lead_and_climax_dominates():
    """Test that the pioneer Keora leads early and the climax Sundari dominates the mature forest."""
    trajectory = _run_until(mangrove_succession(), 400, dt=1.0)
    early, late = 10, -1
    assert trajectory['keora.volume'][early] > trajectory['sundari.volume'][early]
    assert trajectory['sundari.volume'][late] > max(trajectory['keora.volume'][late], trajectory['gewa.volume'][late])
    assert min(trajectory[f'{name}.volume'].min() for name in ('keora', 'gewa', 'sundari')) >= -1e-8
