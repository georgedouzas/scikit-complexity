"""Test the `_brock_hommes` module of `skcomplex.systems.economics.markets`."""

import numpy as np
from scipy.special import expit

from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.economics.markets import BrockHommesSystem


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
        system = BrockHommesSystem(200, intensity=intensity).prepare().system_
        deviations = Simulation(system, random_state=0).prepare().run(until=120).trajectory_.values('pricing.deviation')
        spreads[intensity] = (deviations[-50:].std(), _mean_field(intensity, 400)[-200:].std())
    assert max(spreads[1.0]) < 0.1
    assert min(spreads[6.0]) > 0.3
