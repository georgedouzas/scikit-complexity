"""Test the `_wolf_sheep` module of `skcomplex.systems.biology.ecology`."""

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.ecology import WolfSheepSystem


def test_wolf_sheep_populations_change():
    """Test that the wolf and sheep counts change over 100 steps and are reported every frame."""
    trajectory = simulate(WolfSheepSystem(20, 20, random_state=0), random_state=0, until=100)
    for group in ('sheep', 'wolves'):
        counts = trajectory.values(f'{group}.count')
        assert len(counts) == 101
        assert len(set(counts.tolist())) > 1
    assert set(trajectory.values('grass.count').tolist()) == {400.0}
