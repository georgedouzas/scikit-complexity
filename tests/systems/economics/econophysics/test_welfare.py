"""Test the `_welfare` module of `skcomplex.systems.economics.econophysics`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import NonnegativeElement
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import KineticSolver
from skcomplex.systems.biology.oncology import ImmuneCompetitionSystem
from skcomplex.systems.economics.econophysics import KineticWealthWelfareSystem, WealthDensityElement
from skcomplex.systems.social.crowds import EvacuationSystem
from skcomplex.systems.social.opinion_dynamics import KineticOpinionSystem


def test_welfare_policy_lowers_the_gini_coefficient():
    """Test that a policy moving activity from the richest to the poorest classes lowers the stationary Gini."""
    without = simulate(KineticWealthWelfareSystem(), until=200, dt=50.0, random_state=0)
    with_policy = simulate(KineticWealthWelfareSystem(tax=0.2), until=200, dt=50.0, random_state=0)
    assert with_policy.values('gini')[-1] < without.values('gini')[-1] - 0.05
    assert with_policy.values('gini')[-1] == pytest.approx(with_policy.values('gini')[-2], abs=1e-4)
    np.testing.assert_allclose(without.values('density'), 1.0, rtol=1e-10)


def test_wealth_gini_matches_its_definition():
    """Test that the Gini observable equals the mean absolute difference of wealth over twice the mean."""
    trajectory = simulate(KineticWealthWelfareSystem(), until=20, dt=20.0, random_state=0)
    f = np.asarray(trajectory.values('wealth.x')[-1])
    wealth = np.linspace(0.0, 1.0, f.size)
    expected = (f[:, None] * f[None, :] * np.abs(wealth[:, None] - wealth[None, :])).sum() / (
        2 * f.sum() * (f @ wealth)
    )
    assert trajectory.values('gini')[-1] == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    'system',
    [KineticWealthWelfareSystem, ImmuneCompetitionSystem, KineticOpinionSystem, EvacuationSystem],
)
def test_kinetic_systems_run_with_the_particle_solver(system):
    """Test that the kinetic systems of encounters alone validate and start with the particle solver."""
    trajectory = simulate(system().prepare().system_, until=0, solver=KineticSolver(), random_state=0)
    assert trajectory.times.tolist() == [0.0]


def test_wealth_distribution_holds_wealth_densities():
    """Test that the wealth distribution is a group of wealth densities, a realization of the core atom."""
    wealth = KineticWealthWelfareSystem().prepare().system_.elements[0]
    assert wealth.element is WealthDensityElement
    assert issubclass(WealthDensityElement, NonnegativeElement)
