"""Test the `_wolf_sheep` module of `skcomplex.models.biology.ecology`."""

from skcomplex.core.runtime import Simulation
from skcomplex.models.biology.ecology import wolf_sheep


def test_wolf_sheep_populations_change():
    """Test that the wolf and sheep counts change over 100 steps and are reported every frame."""
    trajectory = Simulation(wolf_sheep(20, 20, random_state=0), random_state=0).run(until=100).trajectory_
    for group in ('sheep', 'wolves'):
        counts = trajectory[f'{group}.count']
        assert len(counts) == 101
        assert len(set(counts.tolist())) > 1
    assert set(trajectory['grass.count'].tolist()) == {400.0}
