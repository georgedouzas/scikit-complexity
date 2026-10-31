"""Test the `_ornstein_uhlenbeck` module of `skcomplex.core.interactions`."""

import numpy as np
import pytest

from skcomplex.core.elements import GroupElement
from skcomplex.core.interactions import OrnsteinUhlenbeckInteraction
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import EulerMaruyamaSolver
from skcomplex.errors import ValidationError

from ...conftest import HoldingElement

MEMBERS = 20_000


def _run(interaction, n, until):
    """Return the values of the state `n` of a group driven by one Ornstein-Uhlenbeck interaction."""
    system = System(
        elements=[GroupElement('walkers', element=HoldingElement, size=MEMBERS, values={'n': n})],
        interactions=[interaction],
    )
    simulation = Simulation(system, solver=EulerMaruyamaSolver(dt=0.005), dt=0.5, random_state=0)
    return simulation.check().prepare().run(until=until).trajectory_.values('walkers.n')


def test_ornstein_uhlenbeck_reaches_its_stationary_variance():
    """Test that the stationary variance is `sigma**2 / (2 theta)` within 5 percent, around the mean."""
    process = OrnsteinUhlenbeckInteraction('ou', target='walkers', state='n', theta=2.0, mean=1.0, sigma=0.6)
    values = _run(process, 1.0, until=6)[-4:]
    assert values.var(axis=1).mean() == pytest.approx(0.6**2 / (2 * 2.0), rel=0.05)
    assert values.mean() == pytest.approx(1.0, abs=0.01)


def test_ornstein_uhlenbeck_multiplicative_gives_the_log_normal_mean():
    """Test that geometric Brownian motion has the mean `x0 exp(mu t)` and the log mean `(mu - sigma**2 / 2) t`."""
    mu, sigma, until = 0.1, 0.2, 1.0
    process = OrnsteinUhlenbeckInteraction(
        'gbm',
        target='walkers',
        state='n',
        theta=-mu,
        mean=0.0,
        sigma=sigma,
        noise='multiplicative',
    )
    final = _run(process, 1.0, until=until)[-1]
    assert final.mean() == pytest.approx(np.exp(mu * until), rel=0.01)
    assert np.log(final).mean() == pytest.approx((mu - sigma**2 / 2) * until, abs=0.005)


def test_check_rejects_an_unknown_noise_form():
    """Test that `check` rejects a noise form other than additive or multiplicative."""
    with pytest.raises(ValidationError, match='noise form'):
        OrnsteinUhlenbeckInteraction('ou', target='walkers', noise='colored').check()
