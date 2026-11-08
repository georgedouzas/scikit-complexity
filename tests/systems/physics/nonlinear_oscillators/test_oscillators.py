"""Test the `_oscillators` module of `skcomplex.systems.physics.nonlinear_oscillators`."""

import math

import numpy as np
import pytest

from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.systems.physics.nonlinear_oscillators import (
    AdlerEquationSystem,
    ForcedTodaSystem,
    ForcedVanDerPolSystem,
    HenonHeilesSystem,
    ImpulsiveOscillatorSystem,
    NuclearQuadrupoleSystem,
    StuartLandauSystem,
    VanDerPolElement,
    VanDerPolSystem,
)

from ..conftest import _run, _series, _upward_crossings


def _run_van_der_pol(**parameters):
    """Return the trajectory of one Van der Pol oscillator over a long run."""
    system = System(elements=[VanDerPolElement('oscillator', **parameters)])
    solver = ScipyODESolver(rtol=1e-9, atol=1e-11)
    return simulate(system, until=100.0, dt=0.05, solver=solver)


def test_unforced_oscillator_reaches_its_limit_cycle():
    """Test that a small displacement grows to the limit cycle of amplitude near 2."""
    trajectory = _run_van_der_pol(x=0.1, mu=1.0)
    assert np.abs(trajectory.values('oscillator.x')[-400:]).max() == pytest.approx(2.0, abs=0.02)


def test_undamped_forced_oscillator_follows_the_forcing():
    """Test that without damping and from rest the forced oscillator follows `A (cos(w t) - cos(t)) / (1 - w^2)`."""
    trajectory = _run_van_der_pol(x=0.0, mu=0.0, A=0.5, omega=2.0)
    expected = 0.5 * (np.cos(2.0 * trajectory.times) - np.cos(trajectory.times)) / (1 - 4.0)
    np.testing.assert_allclose(trajectory.values('oscillator.x'), expected, atol=1e-6)


def test_van_der_pol_reaches_limit_cycle():
    """Test that the Van der Pol oscillator settles on a cycle of amplitude near 2."""
    x = simulate(VanDerPolSystem(), random_state=0).values('oscillator.x')
    assert 1.9 < np.abs(x[-200:]).max() < 2.1


def test_henon_heiles_conserves_energy():
    """Test that the Hénon-Heiles star keeps its energy within its invariant tolerance."""
    trajectory = simulate(HenonHeilesSystem(), random_state=0)
    x, y, px, py = (trajectory.values(name) for name in ('star.x', 'star.y', 'star.px', 'star.py'))
    energy = (px**2 + py**2 + x**2 + y**2) / 2 + x**2 * y - y**3 / 3
    assert np.abs(energy - energy[0]).max() < 1e-6
    np.testing.assert_allclose(trajectory.values('star.energy'), energy)


def test_henon_heiles_stays_bounded_below_the_escape_energy():
    """Test that the star started at the energy 1/12 stays bounded to time 1000 within its energy tolerance."""
    y, py = 0.0, 0.0
    px = math.sqrt(2 / 12)
    trajectory = simulate(HenonHeilesSystem(x=0.0, y=y, px=px, py=py), until=1000.0, random_state=0)
    x, y = (np.asarray(trajectory.values(name)) for name in ('star.x', 'star.y'))
    energy = np.asarray(trajectory.values('star.energy'))
    assert energy[0] == pytest.approx(1 / 12)
    assert np.hypot(x, y).max() < 1.0
    assert np.ptp(energy) < 1e-6


def test_stuart_landau_settles_on_its_limit_cycle():
    """Test that the Stuart-Landau oscillator settles on the circle of radius `sqrt(mu)`."""
    trajectory = _run(StuartLandauSystem(mu=0.5, beta=0.5), 40.0, 0.1)
    x, y = _series(trajectory, 'oscillator.x', 'oscillator.y')
    phase = np.unwrap(np.arctan2(y, x))
    assert np.hypot(x, y)[-1] == pytest.approx(math.sqrt(0.5), rel=1e-6)
    assert np.diff(phase[-20:]).mean() / 0.1 == pytest.approx(1.0 - 0.25, rel=1e-4)


def test_forced_van_der_pol_locks_to_a_strong_force():
    """Test that a strong force entrains the Van der Pol oscillator to its own frequency."""
    trajectory = _run(ForcedVanDerPolSystem(A=1.0, omega=1.1), 120.0, 0.05, solver=ScipyODESolver(rtol=1e-8))
    crossings = _upward_crossings(trajectory.times, _series(trajectory, 'oscillator.x')[0])
    assert np.diff(crossings[-5:]).mean() == pytest.approx(2 * math.pi / 1.1, rel=1e-3)


@pytest.mark.parametrize('detuning', [0.5, 1.5])
def test_adler_equation_locks_inside_the_tongue_and_slips_outside(detuning):
    """Test that the phase difference locks for `|Δω| < ε` and slips with period `2π / sqrt(Δω² - ε²)` beyond."""
    trajectory = _run(AdlerEquationSystem(detuning=detuning), 60.0, 0.05)
    phi = _series(trajectory, 'difference.phi')[0]
    if detuning < 1:
        assert math.sin(phi[-1]) == pytest.approx(-detuning, abs=1e-6)
        return
    slips = np.interp(2 * math.pi * np.arange(1, 8), phi - phi[0], trajectory.times)
    assert np.diff(slips).mean() == pytest.approx(2 * math.pi / math.sqrt(detuning**2 - 1), rel=1e-3)


def test_nuclear_quadrupole_conserves_energy():
    """Test that the quadrupole Hamiltonian stays within its invariant tolerance along a chaotic orbit."""
    trajectory = simulate(NuclearQuadrupoleSystem(), random_state=0)
    q1, q2, p1, p2 = (np.asarray(trajectory.values(f'nucleus.{name}')) for name in ('q1', 'q2', 'p1', 'p2'))
    q = q1**2 + q2**2
    energy = (q + p1**2 + p2**2) / 2 + 0.5 * q1 * (3 * q2**2 - q1**2) / math.sqrt(2) + q**2 / 8
    assert energy[0] == pytest.approx(0.75)
    assert np.abs(energy - energy[0]).max() < 1e-6
    np.testing.assert_allclose(trajectory.values('nucleus.energy'), energy)


def test_free_toda_oscillator_conserves_energy_and_swings_further_on_the_soft_side():
    """Test that the free Toda oscillator keeps `v² / 2 + e^x - 1 - x` and reaches the turning points of energy 1."""
    trajectory = _run(ForcedTodaSystem(x=0.0, v=math.sqrt(2.0), d=0.0, a=0.0), 30.0, 0.01)
    x, v = _series(trajectory, 'toda.x', 'toda.x_d1')
    energy = v**2 / 2 + np.exp(x) - 1 - x
    assert np.ptp(energy) < 1e-8
    assert x.max() == pytest.approx(1.1462, abs=2e-3)
    assert x.min() == pytest.approx(-1.8414, abs=2e-3)


def test_forced_toda_oscillator_shifts_its_mean_to_the_soft_side():
    """Test that the driven Toda oscillator oscillates around a negative mean displacement."""
    trajectory = _run(ForcedTodaSystem(omega=0.8), 300.0, 0.05)
    x = _series(trajectory, 'toda.x')[0]
    assert x[len(x) // 2 :].mean() < -0.05


def test_impulsive_oscillator_follows_the_exact_periodic_orbit():
    """Test that the default kicked orbit repeats with the period `4 a` and the amplitude `arcsin(sin A sin(a / q))`."""
    amplitude, a = 1.2, 0.4165
    q = math.cos(amplitude)
    trajectory = _run(ImpulsiveOscillatorSystem(), 3 * 4 * a, 0.001)
    x, v = _series(trajectory, 'oscillator.x', 'oscillator.x_d1')
    times = np.asarray(trajectory.times)
    assert x.max() == pytest.approx(math.asin(math.sin(amplitude) * math.sin(a / q)), abs=3e-3)
    for period in (1, 2, 3):
        index = np.argmin(np.abs(times - 4 * a * period))
        assert x[index] == pytest.approx(0.0, abs=0.03)
        assert v[index] == pytest.approx(math.tan(amplitude), abs=0.01)


def test_impulsive_oscillator_shifts_and_scales_the_dipole_impulses():
    """Test that the shift moves the impulses to `(1 - gamma) a` and `(3 + gamma) a` and scales them by `1 - gamma²`."""
    trajectory = _run(ImpulsiveOscillatorSystem(a=0.5, gamma=0.5, spring='linear'), 1.8, 0.01)
    pull, push = trajectory.events[:2]
    assert (pull.label, push.label) == ('pull', 'push')
    assert pull.time == pytest.approx(0.25)
    assert push.time == pytest.approx(1.75)
    assert pull.after[1] - pull.before[1] == pytest.approx(-4 / 0.75)
    assert push.after[1] - push.before[1] == pytest.approx(4 / 0.75)
