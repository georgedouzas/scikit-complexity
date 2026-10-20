"""Test the kinetic exchange, minority game, El Farol, Kolkata restaurant, and Axelrod models."""

import numpy as np
import pytest

from skcomplex.core.runtime import Simulation
from skcomplex.models.economics.econophysics import (
    el_farol,
    kinetic_wealth_exchange,
    kolkata_restaurant,
    minority_game,
)
from skcomplex.models.social.opinion_dynamics import axelrod, cultural_traits, largest_region


def _final_wealth(until, **options):
    """Return the wealth of every trader at the end of a kinetic exchange run."""
    trajectory = Simulation(kinetic_wealth_exchange(500, **options), random_state=0).run(until=until).trajectory_
    return trajectory.agents('traders', until).values['wealth']


def test_kinetic_exchange_stationary_distributions():
    """Without saving wealth is exponential, with saving it is Gamma of shape `1 + 3 lambda / (1 - lambda)`."""
    exponential = _final_wealth(150)
    assert exponential.mean() == pytest.approx(1.0)
    assert exponential.var() == pytest.approx(1.0, rel=0.2)
    assert np.mean(exponential < np.log(2)) == pytest.approx(0.5, abs=0.06)
    saving = 0.5
    shape = 1 + 3 * saving / (1 - saving)
    gamma = _final_wealth(150, saving=saving)
    assert gamma.var() == pytest.approx(1 / shape, rel=0.2)
    assert np.mean(gamma < 0.1) < 0.01


def test_kinetic_exchange_debt_shifts_exponential():
    """With a debt limit the wealth is exponential shifted to minus the limit, with temperature `<w> + d`."""
    wealth = _final_wealth(300, debt=1.0, amount=0.25)
    assert wealth.min() >= -1.0
    temperature = 2.0
    assert np.mean(wealth < 0) == pytest.approx(1 - np.exp(-1 / temperature), abs=0.07)
    assert wealth.std() == pytest.approx(temperature, rel=0.2)


def _volatility(memory, n_agents=51, **options):
    """Return `sigma^2 / N` of the attendance of a minority game after a transient."""
    system = minority_game(n_agents, memory=memory, random_state=0, **options)
    attendance = Simulation(system, random_state=0).run(until=400).trajectory_['game.attendance']
    return attendance[100:].var() / n_agents


def test_minority_game_volatility_minimum():
    """The volatility exceeds random play for small `alpha` and falls well below it near `alpha_c ~ 0.34`."""
    assert _volatility(1) > 1
    assert _volatility(4) < 0.6
    assert _volatility(4, evolutionary=True, period=50) < 1


def test_el_farol_attendance_near_capacity():
    """With 100 patrons and capacity 60, the mean attendance after the transient lies within 5 of 60."""
    attendance = Simulation(el_farol(100, capacity=60, random_state=0), random_state=0).run(500).trajectory_
    assert abs(np.mean(attendance['bar.attendance'][100:]) - 60) < 5


def test_kolkata_utilization():
    """Random choice uses `1 - 1/e` of the restaurants and stochastic crowd avoidance about 0.8."""
    utilization = {}
    for strategy in ('random', 'crowd_avoiding'):
        system = kolkata_restaurant(200, strategy=strategy, random_state=0)
        values = Simulation(system, random_state=0).run(until=60).trajectory_['restaurants.utilization']
        utilization[strategy] = np.mean(values[20:])
    assert utilization['random'] == pytest.approx(1 - 1 / np.e, abs=0.02)
    assert utilization['crowd_avoiding'] == pytest.approx(0.8, abs=0.04)


def test_axelrod_order_disorder_transition():
    """For few traits one culture spans most of the lattice, and for many traits small regions freeze."""
    regions = {}
    for traits in (2, 50):
        system = axelrod(8, 8, features=3, traits=traits, random_state=0)
        trajectory = Simulation(system, random_state=0).run(until=300).trajectory_
        regions[traits] = trajectory['census.largest_region'][-1]
    assert regions[2] > 0.5
    assert regions[50] < 0.1


def test_axelrod_helpers():
    """Culture codes decode into base-`q` digits, and regions are connected cells of one culture."""
    assert cultural_traits(2 + 3 * 5, 2, 5).tolist() == [2, 3]
    assert largest_region([(0, 0), (1, 0), (0, 1), (1, 1)], [4, 7, 7, 4], 2, 2) == 0.25
