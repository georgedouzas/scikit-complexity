"""Test the `_welfare` module of `skcomplex.systems.economics.econophysics`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import KineticSolver
from skcomplex.core.systems import System
from skcomplex.systems.biology.oncology import ImmuneCompetitionSystem
from skcomplex.systems.economics.econophysics import KineticWealthWelfareSystem, WealthDensityElement
from skcomplex.systems.general.elements import NonnegativeElement
from skcomplex.systems.social.crowds import EvacuationSystem
from skcomplex.systems.social.opinion_dynamics import KineticOpinionSystem


def test_welfare_policy_lowers_the_gini_coefficient():
    """Test that a policy moving activity from the richest to the poorest classes lowers the stationary Gini."""
    without = KineticWealthWelfareSystem().prepare().simulate(until=200, dt=50.0, random_state=0)
    with_policy = KineticWealthWelfareSystem(tax=0.2).prepare().simulate(until=200, dt=50.0, random_state=0)
    assert with_policy.get_values('gini')[-1] < without.get_values('gini')[-1] - 0.05
    assert with_policy.get_values('gini')[-1] == pytest.approx(with_policy.get_values('gini')[-2], abs=1e-4)
    np.testing.assert_allclose(without.get_values('density'), 1.0, rtol=1e-10)


def test_wealth_gini_matches_its_definition():
    """Test that the Gini observable equals the mean absolute difference of wealth over twice the mean."""
    trajectory = KineticWealthWelfareSystem().prepare().simulate(until=20, dt=20.0, random_state=0)
    f = np.asarray(trajectory.get_values('wealth.x')[-1])
    wealth = np.linspace(0.0, 1.0, f.size)
    expected = (f[:, None] * f[None, :] * np.abs(wealth[:, None] - wealth[None, :])).sum() / (
        2 * f.sum() * (f @ wealth)
    )
    assert trajectory.get_values('gini')[-1] == pytest.approx(expected, rel=1e-9)


def test_kinetic_wealth_runs_with_the_particle_solver():
    """Test that the kinetic wealth system of encounters alone validates and starts with the particle solver."""
    system = KineticWealthWelfareSystem(solver=KineticSolver()).prepare()
    assert system.simulate(until=0, random_state=0).times.tolist() == [0.0]


@pytest.mark.parametrize('system', [ImmuneCompetitionSystem, KineticOpinionSystem, EvacuationSystem])
def test_kinetic_systems_run_with_the_particle_solver(system):
    """Test that the kinetic systems of encounters alone validate and start with the particle solver."""
    built = system().prepare().system_
    if isinstance(built, System):
        trajectory = built.set_params(solver=KineticSolver()).prepare().simulate(until=0, random_state=0)
    else:
        trajectory = simulate(built, until=0, solver=KineticSolver(), random_state=0)
    assert trajectory.times.tolist() == [0.0]


def test_wealth_distribution_holds_wealth_densities():
    """Test that the wealth distribution is a wealth density with one value per class."""
    wealth = KineticWealthWelfareSystem().prepare().system_.elements[0]
    assert isinstance(wealth, WealthDensityElement)
    assert issubclass(WealthDensityElement, NonnegativeElement)
