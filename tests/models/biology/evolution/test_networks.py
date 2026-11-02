"""Test the `_networks` module of `skcomplex.models.biology.evolution`."""

import numpy as np

from ..conftest import _run


def test_sole_manrubia_keeps_its_species_and_extinguishes_in_avalanches():
    """Test that extinct species are replaced, and extinctions come in avalanches of several species."""
    trajectory = _run('sole_manrubia', 200, 1.0)
    sizes = np.nansum(trajectory.values('species.age')[1:] == 0, axis=1)
    assert all(trajectory.network(frame).nodes.size == 100 for frame in (0, 100, 200))
    assert sizes.sum() > 50
    assert sizes[100:].max() > 1
    assert np.count_nonzero(sizes[100:] == 0) > 10


def test_jain_krishna_grows_an_autocatalytic_set_that_spans_the_species():
    """Test that the populated species jump from a few to nearly all once an autocatalytic set forms."""
    trajectory = _run('jain_krishna', 1500, 5.0)
    populated = np.nansum(trajectory.values('species.x') > 1e-3, axis=1)
    links = [trajectory.network(frame).sources.size for frame in (1, len(trajectory.times) - 1)]
    assert populated[1:40].min() < 20
    assert populated[-1] >= 35
    assert links[1] > links[0]
