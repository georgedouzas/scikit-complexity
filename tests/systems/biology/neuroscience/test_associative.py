"""Test the `_associative` module of `skcomplex.systems.biology.neuroscience`."""

import numpy as np

from skcomplex.systems.biology.neuroscience import AssociativeThoughtProcessSystem
from skcomplex.systems.biology.neuroscience._associative import CLIQUES

from ..conftest import _run_until


def test_activity_hops_between_the_stored_cliques():
    """Test Gros's transient state dynamics: the lasting coalitions are the stored cliques, visited in turn."""
    activities = _run_until(AssociativeThoughtProcessSystem(), 1500, dt=2.0).get_values('sites.x')
    coalitions = [tuple(np.flatnonzero(row > 0.85)) for row in activities]
    winners = [coalition for coalition in set(coalitions) if len(coalition) > 1 and coalitions.count(coalition) > 20]
    assert {tuple(clique) for clique in CLIQUES} >= set(winners)
    assert len(set(winners)) >= 4


def test_without_reservoir_dynamics_the_first_clique_stays():
    """Test that without depletion of the reservoirs the cliques are stable attractors."""
    activities = _run_until(AssociativeThoughtProcessSystem(gamma_minus=0.0), 1500, dt=2.0).get_values('sites.x')
    np.testing.assert_allclose(activities[-1], [1, 1, 0, 0, 0, 0, 0], atol=1e-3)
