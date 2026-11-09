"""Test the `_standardization` module of `skcomplex.systems.social.opinion_dynamics`."""

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import gammaln

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.social.opinion_dynamics import StandardizedAgentsSystem


def test_standardized_agents_match_the_exact_equilibrium():
    """Test that the time-averaged share q of ten agents matches the exact distribution P ~ exp(E / T)."""
    size, configurations, K = 10, 20, 4.0
    temperature = K / (2 * np.log(configurations / 2))
    marked = np.arange(size + 1)
    weights = gammaln(size + 1) - gammaln(marked + 1) - gammaln(size - marked + 1)
    weights += marked * np.log(2) + (size - marked) * np.log(configurations)
    weights += K * marked * (marked - 1) / (2 * size * temperature)
    probabilities = np.exp(weights - weights.max()) / np.exp(weights - weights.max()).sum()
    system = (
        StandardizedAgentsSystem(size, K=K, J=0.0, T=temperature, H=configurations, standardized=0.0, random_state=0)
        .prepare()
        .system_
    )
    q = simulate(
        system,
        until=5000,
        random_state=0,
        observables=StandardizedAgentsSystem().get_run_settings()['observables'],
    ).get_values('q')
    assert q[200:].mean() == pytest.approx(np.sum(probabilities * marked / size), abs=0.04)


@pytest.mark.parametrize(('temperature', 'ordered'), [(0.5, True), (1.3, False)])
def test_standardized_agents_order_below_t_m(temperature, ordered):
    """Test that standardized agents magnetize below T_M = J, to the root of m = tanh(J m / T), and not above."""
    system = StandardizedAgentsSystem(100, K=10.0, J=1.0, T=temperature, H=50, random_state=0).prepare().system_
    trajectory = simulate(
        system,
        until=1000,
        random_state=0,
        observables=StandardizedAgentsSystem().get_run_settings()['observables'],
    )
    magnetization = np.abs(trajectory.get_values('m')[-200:]).mean()
    assert trajectory.get_values('q')[-200:].mean() > 0.95
    if ordered:
        assert magnetization == pytest.approx(brentq(lambda m: m - np.tanh(m / temperature), 0.1, 1.0), abs=0.08)
    else:
        assert magnetization < 0.2
