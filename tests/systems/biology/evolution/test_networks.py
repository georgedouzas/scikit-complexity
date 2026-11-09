"""Test the `_networks` module of `skcomplex.systems.biology.evolution`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.evolution import CompetitiveJainKrishnaSystem, JainKrishnaSystem, SoleManrubiaSystem

from ..conftest import _run


def test_sole_manrubia_keeps_its_species_and_extinguishes_in_avalanches():
    """Test that extinct species are replaced, and extinctions come in avalanches of several species."""
    trajectory = _run(SoleManrubiaSystem(), 200, 1.0)
    sizes = np.nansum(trajectory.values('species.age')[1:] == 0, axis=1)
    assert all(trajectory.network(frame).nodes.size == 100 for frame in (0, 100, 200))
    assert sizes.sum() > 50
    assert sizes[100:].max() > 1
    assert np.count_nonzero(sizes[100:] == 0) > 10


def test_jain_krishna_grows_an_autocatalytic_set_that_spans_the_species():
    """Test that the populated species jump from a few to nearly all once an autocatalytic set forms."""
    trajectory = _run(JainKrishnaSystem(), 1500, 5.0)
    populated = np.nansum(trajectory.values('species.x') > 1e-3, axis=1)
    links = [trajectory.network(frame).sources.size for frame in (1, len(trajectory.times) - 1)]
    assert populated[1:40].min() < 20
    assert populated[-1] >= 35
    assert links[1] > links[0]


def test_competition_selects_cooperating_species():
    """Test Jain and Krishna with competition: the share of positive links grows from about a half."""

    def positive_share(links: np.ndarray) -> float:
        matrix = np.array(links, dtype=float)
        np.fill_diagonal(matrix, 0.0)
        return float((matrix[matrix != 0] > 0).mean())

    trajectory = simulate(CompetitiveJainKrishnaSystem(), random_state=0, until=300)
    assert positive_share(trajectory.get_agents('species', 0).values['links']) < 0.55
    assert (
        np.mean([positive_share(trajectory.get_agents('species', frame).values['links']) for frame in (200, 250, 300)])
        > 0.65
    )
