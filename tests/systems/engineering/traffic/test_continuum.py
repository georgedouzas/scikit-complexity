"""Test the `_continuum` module of `skcomplex.systems.engineering.traffic`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import Simulation, simulate
from skcomplex.core.engine.solvers import SSPRKSolver
from skcomplex.systems.engineering.traffic import (
    GktTrafficSystem,
    KernerKonhauserSystem,
    LighthillWhithamRichardsSystem,
    PayneSystem,
)


def test_lighthill_whitham_richards_shock_speed():
    """Test that the shock between free and queued traffic moves at the Rankine-Hugoniot speed."""
    trajectory = (
        Simulation(LighthillWhithamRichardsSystem().prepare().system_, dt=0.05, solver=SSPRKSolver())
        .prepare()
        .run(until=1.0)
        .trajectory_
    )
    density = trajectory.values('road.rho')[-1]
    centers = (np.arange(200) + 0.5) / 200
    assert centers[np.argmax(np.diff(density))] == pytest.approx(0.5 + (1 - 1.1), abs=0.01)


def test_payne_conserves_cars_and_keeps_equilibrium():
    """Test that the Payne system conserves the cars and keeps uniform flow at the equilibrium speed."""
    trajectory = (
        Simulation(PayneSystem(50).prepare().system_, dt=0.1, solver=SSPRKSolver()).prepare().run(until=0.5).trajectory_
    )
    assert trajectory.values('road.rho').sum(axis=1) == pytest.approx(
        np.full(6, trajectory.values('road.rho')[0].sum()),
    )
    uniform = (
        Simulation(PayneSystem(50, bump=0.0).prepare().system_, dt=0.1, solver=SSPRKSolver())
        .prepare()
        .run(until=0.5)
        .trajectory_
    )
    assert np.ptp(uniform.values('road.q')[-1]) < 1e-12


def test_kerner_konhauser_forms_a_cluster():
    """Test that a small bump of density grows into a dense cluster while the cars are conserved."""
    trajectory = (
        Simulation(KernerKonhauserSystem().prepare().system_, dt=10.0, solver=SSPRKSolver(cfl=0.15))
        .prepare()
        .run(until=40)
        .trajectory_
    )
    density = trajectory.values('road.rho')
    assert density[-1].max() > 0.35
    assert density.sum(axis=1)[-1] == pytest.approx(density.sum(axis=1)[0])


def test_gkt_traffic_forms_stop_and_go_waves_at_unstable_densities():
    """Test that a perturbation decays in light traffic and grows into clusters with upstream-moving fronts."""
    light = np.asarray(
        simulate(GktTrafficSystem(density=15.0, amplitude=10.0)).values('road.rho'),
    )
    assert np.ptp(light[-1]) < 0.5 * np.ptp(light[0])
    density = np.asarray(simulate(GktTrafficSystem()).values('road.rho'))
    assert np.ptp(density[-1]) > 20.0
    assert density.sum(axis=1)[-1] == pytest.approx(density.sum(axis=1)[0])
    spacing, lag = 10.0 / 200, 10
    shifts = []
    for earlier, later in zip(density[30:-lag], density[30 + lag :], strict=True):
        correlation = np.fft.irfft(np.conj(np.fft.rfft(earlier - earlier.mean())) * np.fft.rfft(later - later.mean()))
        shift = int(np.argmax(correlation))
        shifts.append((shift - 200 if shift > 100 else shift) * spacing)
    velocity = np.mean(shifts) / (lag / 60)
    assert -25.0 <= velocity <= -10.0
