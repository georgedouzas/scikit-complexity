"""Test the `_standardization` module of `skcomplex.models.social.opinion_dynamics`."""

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import gammaln

from skcomplex.core.runtime import Simulation
from skcomplex.models import describe
from skcomplex.models.social.opinion_dynamics import standardized_agents


def test_standardized_agents_match_the_exact_equilibrium():
    """Test that the time-averaged share q of ten agents matches the exact distribution P ~ exp(E / T)."""
    size, configurations, K = 10, 20, 4.0
    temperature = K / (2 * np.log(configurations / 2))
    marked = np.arange(size + 1)
    weights = gammaln(size + 1) - gammaln(marked + 1) - gammaln(size - marked + 1)
    weights += marked * np.log(2) + (size - marked) * np.log(configurations)
    weights += K * marked * (marked - 1) / (2 * size * temperature)
    probabilities = np.exp(weights - weights.max()) / np.exp(weights - weights.max()).sum()
    system = standardized_agents(size, K=K, J=0.0, T=temperature, H=configurations, standardized=0.0, random_state=0)
    q = (
        Simulation(system, random_state=0, observables=describe('standardized_agents').observables_of(system))
        .run(until=5000)
        .trajectory_['q']
    )
    assert q[200:].mean() == pytest.approx(np.sum(probabilities * marked / size), abs=0.04)


@pytest.mark.parametrize(('temperature', 'ordered'), [(0.5, True), (1.3, False)])
def test_standardized_agents_order_below_t_m(temperature, ordered):
    """Test that standardized agents magnetize below T_M = J, to the root of m = tanh(J m / T), and not above."""
    system = standardized_agents(100, K=10.0, J=1.0, T=temperature, H=50, random_state=0)
    trajectory = (
        Simulation(system, random_state=0, observables=describe('standardized_agents').observables_of(system))
        .run(until=1000)
        .trajectory_
    )
    magnetization = np.abs(trajectory['m'][-200:]).mean()
    assert trajectory['q'][-200:].mean() > 0.95
    if ordered:
        assert magnetization == pytest.approx(brentq(lambda m: m - np.tanh(m / temperature), 0.1, 1.0), abs=0.08)
    else:
        assert magnetization < 0.2
