"""Test the `_microbial_communities` module of `skcomplex.models.biology.ecology`."""

import numpy as np

from skcomplex.models.biology.ecology import generalized_lotka_volterra

from ..conftest import SYSTEMS_PRECISE, _run_with_solver


def test_generalized_lotka_volterra_reaches_coexistence():
    """Test that the community settles on the interior fixed point `x = -a^-1 b`."""
    b = np.array([1.0, 0.8, 0.6])
    a = np.array([[-1.0, -0.3, 0.2], [-0.2, -1.0, -0.3], [0.1, -0.2, -1.0]])
    trajectory = _run_with_solver(
        generalized_lotka_volterra(),
        until=60.0,
        dt=0.5,
        solver=SYSTEMS_PRECISE,
    )
    assert np.allclose(trajectory['species.x'][-1], -np.linalg.solve(a, b), rtol=1e-4)
