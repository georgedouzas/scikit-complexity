"""Test the `_power_laws` module of `skcomplex.systems.mathematics.stochastic_processes`."""

import numpy as np
import pytest

from skcomplex.systems import create_simulation


@pytest.mark.parametrize(('mu', 'lam'), [(1.0, 1.5), (2.0, 1.0)])
def test_exponential_growth_at_random_times_has_the_tail_exponent_lam_over_mu(mu, lam):
    """Test that the Hill estimate of the tail of the observed values is `lam / mu`, the power law `x^(-lam/mu - 1)`."""
    simulation = create_simulation('exponential_growth_at_random_times', n_samples=4000, mu=mu, lam=lam)
    x = np.sort(simulation.prepare().run().trajectory_.values('samples.x')[-1])[::-1]
    tail = 400
    assert 1 / np.mean(np.log(x[:tail] / x[tail])) == pytest.approx(lam / mu, rel=0.15)
