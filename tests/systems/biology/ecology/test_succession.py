"""Test the `_succession` module of `skcomplex.systems.biology.ecology`."""

from skcomplex.systems.biology.ecology import (
    MangroveSuccessionSystem,
)

from ..conftest import _run_until


def test_mangrove_pioneers_lead_and_climax_dominates():
    """Test that the pioneer Keora leads early and the climax Sundari dominates the mature forest."""
    trajectory = _run_until(MangroveSuccessionSystem(), 400, dt=1.0)
    early, late = 10, -1
    assert trajectory.get_values('keora.volume')[early] > trajectory.get_values('sundari.volume')[early]
    assert trajectory.get_values('sundari.volume')[late] > max(
        trajectory.get_values('keora.volume')[late],
        trajectory.get_values('gewa.volume')[late],
    )
    assert min(trajectory.get_values(f'{name}.volume').min() for name in ('keora', 'gewa', 'sundari')) >= -1e-8
