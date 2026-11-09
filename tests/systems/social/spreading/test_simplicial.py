"""Test the `_simplicial` module of `skcomplex.systems.social.spreading`."""

import numpy as np

from skcomplex.systems.social.spreading import SimplicialContagionSystem

from ..conftest import _run


def _prevalence(share: float) -> np.ndarray:
    """Return the share of infected nodes at every frame of a run from an initial share."""
    infected = _run(SimplicialContagionSystem(share=share, seed=0), 30.0, dt=0.5).get_values('nodes.infected')
    return infected.reshape(len(infected), -1).mean(axis=1)


def test_simplicial_contagion_is_bistable_above_the_critical_triangle_infectivity():
    """Test that the epidemic dies out from a small seed and stays endemic from a large one, at the same rates."""
    assert _prevalence(0.02)[-1] == 0
    assert _prevalence(0.4)[-10:].mean() > 0.3


def test_simplicial_contagion_infects_only_through_links_without_triangles():
    """Test that without triangles and below the epidemic threshold of the links the epidemic dies out."""
    system = (
        SimplicialContagionSystem(100, mean_degree=6.0, mean_triangles=0.0, infectivity=0.5, share=0.5, seed=1)
        .prepare()
        .system_
    )
    infected = _run(system, 30.0, dt=0.5).get_values('nodes.infected')
    assert infected[-1].sum() == 0
