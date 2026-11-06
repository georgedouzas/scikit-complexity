"""Test the `_factories` module of `skcomplex.systems.biology.epidemiology`."""

import numpy as np
import pytest

from skcomplex.systems.biology.epidemiology import (
    sirs,
    sis,
    zombie_outbreak,
)

from ..conftest import _run_until


@pytest.mark.parametrize('stochastic', [False, True])
def test_sis_settles_at_endemic_equilibrium(stochastic):
    """Test that the SIS system settles at `I = N - gamma / beta`, in mean for random events."""
    trajectory = _run_until(sis(stochastic=stochastic), 300, dt=1.0, random_state=0)
    assert trajectory.values('I.n')[100:].mean() == pytest.approx(1000 - 0.1 / 0.0003, rel=0.05)
    assert trajectory.values('S.n')[-1] + trajectory.values('I.n')[-1] == pytest.approx(1000)


def test_sirs_settles_at_endemic_equilibrium():
    """Test that the SIRS system settles at `S = gamma / beta` and `I = xi (N - S) / (gamma + xi)`."""
    trajectory = _run_until(sirs(), 1000, dt=1.0)
    susceptible = 0.1 / 0.0003
    assert trajectory.values('S.n')[-1] == pytest.approx(susceptible, rel=1e-3)
    assert trajectory.values('I.n')[-1] == pytest.approx(0.02 * (1000 - susceptible) / 0.12, rel=1e-3)


@pytest.mark.parametrize('stochastic', [False, True])
def test_zombies_overrun_humans_and_conserve_the_total(stochastic):
    """Test Munz et al.: without births the total is conserved and the humans vanish, the doomsday state."""
    trajectory = _run_until(zombie_outbreak(zombies=5.0, stochastic=stochastic), 10, dt=0.5, random_state=1)
    total = trajectory.values('S.n') + trajectory.values('Z.n') + trajectory.values('R.n')
    np.testing.assert_allclose(total, 505.0, rtol=1e-9)
    assert trajectory.values('S.n')[-1] < 1.0
    assert trajectory.values('Z.n')[-1] > 200.0
