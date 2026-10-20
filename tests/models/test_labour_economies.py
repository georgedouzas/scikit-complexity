"""Test the labour market, fair food, Sugarscape, and Brock-Hommes models of agent economies."""

import numpy as np
from scipy.special import expit

from skcomplex.core.runtime import Simulation
from skcomplex.models.economics.markets import brock_hommes, fair_food, labour_market, sugarscape
from skcomplex.models.economics.markets._sugarscape import gini


def test_labour_market_flow_balance():
    """Unemployment settles at s / (s + f) and every hire takes an offer of the same step."""
    separation = 0.05
    trajectory = Simulation(labour_market(150, 30, separation=separation), random_state=0).run(until=150).trajectory_
    window = slice(50, None)
    finding = np.nansum(trajectory['labour.volume'][window]) / np.nansum(trajectory['office.searching'][window])
    unemployment = trajectory['unemployment'][window].mean()
    assert 0.05 < finding < 0.95
    assert abs(unemployment - separation / (separation + finding)) < 0.03
    workers = trajectory.agents('workers', 150)
    hired = workers.values['hired'] >= 0
    assert hired.sum() > 50
    np.testing.assert_array_equal(workers.values['hired'][hired], workers.values['offer'][hired])


def _fair_share(**kwargs):
    """Return the mean share of fair food over the last steps of a run."""
    trajectory = Simulation(fair_food(12, 12, **kwargs), random_state=0).run(until=25).trajectory_
    return trajectory['fair_share'][-10:].mean()


def test_fair_food_ethics_price_and_influence():
    """The fair share rises with the ethics weight, falls with the premium, and rises with social influence."""
    base = _fair_share()
    assert _fair_share(ethics_weight=4.0) > base + 0.05
    assert _fair_share(premium=0.8) < base - 0.05
    assert _fair_share(social_weight=0.0) < base - 0.05


def test_sugarscape_carrying_capacity_and_inequality():
    """The population falls to a stable carrying capacity and the wealth becomes unequal."""
    trajectory = Simulation(sugarscape(30, 30, n_foragers=200), random_state=0).run(until=80).trajectory_
    population, inequality = trajectory['census.population'], trajectory['census.gini']
    assert population[-1] < 200
    assert abs(population[-1] - population[-20]) <= 0.05 * population[-1]
    assert inequality[-1] > 0.3
    assert inequality[-1] > gini(trajectory.agents('foragers', 0).values['sugar']) + 0.05
    assert gini([1.0, 1.0, 1.0]) == 0.0


def _mean_field(intensity, steps, *, rate=1.1, strength=1.2, cost=1.0, noise=0.01, initial=0.1):
    """Return the deviations of the Brock-Hommes map with infinitely many traders and the same noise level."""
    rng = np.random.default_rng(1)
    history, previous, fitness, deviations = initial, None, np.zeros(2), []
    for _ in range(steps):
        share = expit(intensity * (fitness[1] - fitness[0]))
        forecasts = np.array([0.0, strength * history])
        price = (forecasts @ [1 - share, share] + rng.normal(0.0, noise)) / rate
        fitness = np.zeros(2) if previous is None else (price - rate * history) * (previous - rate * history)
        fitness = fitness - [cost, 0.0]
        previous, history = forecasts, price
        deviations.append(price)
    return np.array(deviations)


def test_brock_hommes_bifurcation():
    """The price converges to the fundamental for low intensity and fluctuates for high, as the mean-field map."""
    spreads = {}
    for intensity in (1.0, 6.0):
        system = brock_hommes(200, intensity=intensity)
        deviations = Simulation(system, random_state=0).run(until=120).trajectory_['pricing.deviation']
        spreads[intensity] = (deviations[-50:].std(), _mean_field(intensity, 400)[-200:].std())
    assert max(spreads[1.0]) < 0.1
    assert min(spreads[6.0]) > 0.3
