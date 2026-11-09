"""Test the `_factories` module of `skcomplex.systems.biology.epidemiology`."""

import numpy as np
import pytest

from skcomplex.core.components.systems import System
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.types import FormKind
from skcomplex.systems.biology.common.kinetics import CompartmentElement
from skcomplex.systems.biology.epidemiology import (
    InfectionInteraction,
    RecoveryInteraction,
    SIRSSystem,
    SISSystem,
    ZombieOutbreakSystem,
)

from ..conftest import _run_until


@pytest.mark.parametrize('stochastic', [False, True])
def test_sis_settles_at_endemic_equilibrium(stochastic):
    """Test that the SIS system settles at `I = N - gamma / beta`, in mean for random events."""
    trajectory = _run_until(SISSystem(stochastic=stochastic), 300, dt=1.0, random_state=0)
    assert trajectory.values('I.x')[100:].mean() == pytest.approx(1000 - 0.1 / 0.0003, rel=0.05)
    assert trajectory.values('S.x')[-1] + trajectory.values('I.x')[-1] == pytest.approx(1000)


def test_sirs_settles_at_endemic_equilibrium():
    """Test that the SIRS system settles at `S = gamma / beta` and `I = xi (N - S) / (gamma + xi)`."""
    trajectory = _run_until(SIRSSystem(), 1000, dt=1.0)
    susceptible = 0.1 / 0.0003
    assert trajectory.values('S.x')[-1] == pytest.approx(susceptible, rel=1e-3)
    assert trajectory.values('I.x')[-1] == pytest.approx(0.02 * (1000 - susceptible) / 0.12, rel=1e-3)


@pytest.mark.parametrize('stochastic', [False, True])
def test_zombies_overrun_humans_and_conserve_the_total(stochastic):
    """Test Munz et al.: without births the total is conserved and the humans vanish, the doomsday state."""
    trajectory = _run_until(ZombieOutbreakSystem(zombies=5.0, stochastic=stochastic), 10, dt=0.5, random_state=1)
    total = trajectory.values('S.x') + trajectory.values('Z.x') + trajectory.values('R.x')
    np.testing.assert_allclose(total, 505.0, rtol=1e-9)
    assert trajectory.values('S.x')[-1] < 1.0
    assert trajectory.values('Z.x')[-1] > 200.0


def test_epidemic_stochastic_mode_is_jump_process():
    """Test that stochastic infection and recovery give integer counts of a jump process."""
    system = System(
        elements=[CompartmentElement('S', x=990), CompartmentElement('I', x=10), CompartmentElement('R')],
        interactions=[
            InfectionInteraction('infection', susceptible='S', infected='I', beta=0.0005, stochastic=True),
            RecoveryInteraction('recovery', infected='I', recovered='R', gamma=0.1, stochastic=True),
        ],
    )
    assert PassCompiler().prepare(system).run().problem_.form == FormKind.JUMP
    states = simulate(system, dt=1.0, random_state=3, until=50).states
    assert np.all(states >= 0)
    assert np.all(states == np.round(states))
    assert np.allclose(states.sum(axis=1), 1000)


def test_epidemic_deterministic_mode_conserves_population():
    """Test that the deterministic SIR system keeps the total population."""
    system = System(
        elements=[CompartmentElement('S', x=990), CompartmentElement('I', x=10), CompartmentElement('R')],
        interactions=[
            InfectionInteraction('infection', susceptible='S', infected='I', beta=0.0005),
            RecoveryInteraction('recovery', infected='I', recovered='R', gamma=0.1),
        ],
    )
    states = simulate(system, dt=1.0, until=50).states
    assert np.allclose(states.sum(axis=1), 1000)
