"""Test the `_welfare` module of `skcomplex.systems.economics.econophysics`."""

import numpy as np
import pytest

from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.solvers import KineticSolver
from skcomplex.systems import create_system

from ...conftest import _trajectory


def test_welfare_policy_lowers_the_gini_coefficient():
    """Test that a policy moving activity from the richest to the poorest classes lowers the stationary Gini."""
    without = _trajectory('kinetic_wealth_welfare', 200, 50.0)
    with_policy = _trajectory('kinetic_wealth_welfare', 200, 50.0, tax=0.2)
    assert with_policy.values('gini')[-1] < without.values('gini')[-1] - 0.05
    assert with_policy.values('gini')[-1] == pytest.approx(with_policy.values('gini')[-2], abs=1e-4)
    np.testing.assert_allclose(without.values('density'), 1.0, rtol=1e-10)


def test_wealth_gini_matches_its_definition():
    """Test that the Gini observable equals the mean absolute difference of wealth over twice the mean."""
    trajectory = _trajectory('kinetic_wealth_welfare', 20, 20.0)
    f = np.asarray(trajectory.values('wealth.f')[-1])
    wealth = np.linspace(0.0, 1.0, f.size)
    expected = (f[:, None] * f[None, :] * np.abs(wealth[:, None] - wealth[None, :])).sum() / (
        2 * f.sum() * (f @ wealth)
    )
    assert trajectory.values('gini')[-1] == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize('name', ['kinetic_wealth_welfare', 'immune_competition', 'kinetic_opinion', 'evacuation'])
def test_kinetic_systems_run_with_the_particle_solver(name):
    """Test that the kinetic systems of encounters alone validate with the particle solver."""
    assert PassCompiler(solver=KineticSolver()).prepare(create_system(name)).report_.ok
