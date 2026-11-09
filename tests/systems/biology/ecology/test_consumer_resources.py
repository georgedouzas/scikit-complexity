"""Test the `_consumer_resources` module of `skcomplex.systems.biology.ecology`."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from skcomplex.core.errors import ValidationError
from skcomplex.systems.biology.ecology import ConsumerResourceSystem

from ..conftest import _run_precise


def test_macarthur_consumers_coexist_where_their_isoclines_cross():
    """Test that the consumers of MacArthur's model settle where both zero net growth isoclines `c R = m` cross."""
    trajectory = _run_precise(ConsumerResourceSystem(), 200, dt=1.0)
    resources = trajectory.get_values('resources.R')[-1]
    np.testing.assert_allclose(resources, [5 / 3, 5 / 9], rtol=1e-5)
    assert np.all(trajectory.get_values('consumers.N')[-1] > 0.1)


def test_gros_competition_for_one_resource_leaves_the_fastest_consumer():
    """Test Gros's solution: the total tends to `a / d`, the resource to `d / max r`, and one consumer survives."""
    a, d = 2.0, 0.5
    system = (
        ConsumerResourceSystem(
            [0.1, 0.1, 0.1],
            [1.0],
            c=((1.0,), (1.5,), (2.0,)),
            m=d,
            K=1.0,
            kappa=a,
            omega=0.0,
            supply='chemostat',
        )
        .prepare()
        .system_
    )
    trajectory = _run_precise(system, 300, dt=1.0)
    consumers = trajectory.get_values('consumers.N')[-1]
    assert consumers.sum() == pytest.approx(a / d, rel=1e-4)
    assert trajectory.get_values('resources.R')[-1, 0] == pytest.approx(d / 2.0, rel=1e-4)
    assert consumers[:2].max() < 1e-3


def test_essential_resources_follow_their_equations():
    """Test that essential resources in a chemostat follow Eqs.

    (13)-(14) of Mehta and Blumenthal (2024).
    """
    c = np.array([[0.5, 0.3], [0.4, 0.6]])
    v = np.array([[1.0, 0.2], [0.3, 1.0]])
    kappa, n, m = np.array([4.8, 2.85]), -4.0, 0.6

    def rates(t, y):
        consumers, resources = y[:2], y[2:]
        growth = ((c * resources) ** n).sum(axis=1) ** (1 / n)
        return np.r_[consumers * (growth - m), kappa - resources - (v * (consumers * growth)[:, None]).sum(axis=0)]

    reference = solve_ivp(rates, (0, 50), [0.1, 0.1, 4.8, 2.85], rtol=1e-10, atol=1e-12).y[:, -1]
    system = ConsumerResourceSystem(supply='chemostat', kind='essential', n=n, v=v, m=m).prepare().system_
    trajectory = _run_precise(system, 50, dt=1.0)
    np.testing.assert_allclose(trajectory.get_values('consumers.N')[-1], reference[:2], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(trajectory.get_values('resources.R')[-1], reference[2:], rtol=1e-5)


def test_consumer_resource_rejects_mismatched_initial_values():
    """Test that initial values that do not match the uptake matrix raise an error."""
    with pytest.raises(ValidationError):
        ConsumerResourceSystem([0.1, 0.1, 0.1]).prepare()
