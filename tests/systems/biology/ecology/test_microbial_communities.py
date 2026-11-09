"""Test the `_microbial_communities` module of `skcomplex.systems.biology.ecology`."""

import numpy as np
from scipy.integrate import solve_ivp

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.experiments import sweep
from skcomplex.systems.biology.ecology import GeneralizedLotkaVolterraSystem
from skcomplex.systems.biology.ecology._microbial_communities import COMMUNITY_GROWTH, COMMUNITY_INTERACTIONS

from ..conftest import SYSTEMS_PRECISE, _run_with_solver


def test_generalized_lotka_volterra_reaches_coexistence():
    """Test that the community settles on the interior fixed point `x = -a^-1 b`."""
    b = np.array([1.0, 0.8, 0.6])
    a = np.array([[-1.0, -0.3, 0.2], [-0.2, -1.0, -0.3], [0.1, -0.2, -1.0]])
    trajectory = _run_with_solver(
        GeneralizedLotkaVolterraSystem(),
        until=60.0,
        dt=0.5,
        solver=SYSTEMS_PRECISE,
    )
    assert np.allclose(trajectory.get_values('species.x')[-1], -np.linalg.solve(a, b), rtol=1e-4)


def test_generalized_lotka_volterra_keeps_the_trajectory_of_the_constant_matrix():
    """Test that the matrix parameter gives the trajectory of the rate `x_i (b_i + sum_j a_ij x_j)` of NumPy."""
    b, a = np.array(COMMUNITY_GROWTH), np.array(COMMUNITY_INTERACTIONS)
    trajectory = simulate(GeneralizedLotkaVolterraSystem())
    expected = solve_ivp(
        lambda t, x: x * (b + a @ x),
        (0.0, 30.0),
        np.full(3, 0.1),
        t_eval=trajectory.times,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(trajectory.get_values('species.x'), expected.y.T, atol=1e-6)


def test_generalized_lotka_volterra_matrix_swept_by_an_experiment():
    """Test that a grid experiment over two matrices that differ off the diagonal gives different abundances."""
    first = np.array(COMMUNITY_INTERACTIONS)
    second = first.copy()
    second[0, 1] = 0.3
    bounds = {'glv.a': [first, second]}
    result = sweep(GeneralizedLotkaVolterraSystem().prepare().system_, bounds, until=30.0, outputs=['species.x'])
    final = result.series('species.x')[:, -1]
    assert not np.allclose(final[0], final[1])
