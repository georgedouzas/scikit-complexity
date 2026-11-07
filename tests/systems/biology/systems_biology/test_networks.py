"""Test the `_networks` module of `skcomplex.systems.biology.systems_biology`."""

import numpy as np

from skcomplex.systems.biology.systems_biology import (
    BiochemicalSystemsTheorySystem,
    GlycolysisOscillationsSystem,
    GoodwinOscillatorSystem,
    PhosphorylationOscillatorSystem,
)

from ..conftest import SYSTEMS_PRECISE, _run_with_solver


def test_goodwin_oscillator_switches_at_hill_coefficient_eight():
    """Test that three stages settle on the fixed point for a Hill coefficient of 6 and oscillate for 12."""
    amplitudes = {}
    for n in (6.0, 12.0):
        trajectory = _run_with_solver(
            GoodwinOscillatorSystem(3, n=n),
            until=200.0,
            dt=0.1,
            solver=SYSTEMS_PRECISE,
        )
        late = trajectory.values('x3.x')[trajectory.times >= 150.0]
        amplitudes[n] = late.max() - late.min()
    assert amplitudes[6.0] < 1e-4
    assert amplitudes[12.0] > 0.5


def test_phosphorylation_oscillator_cycles():
    """Test that the three switches turn on in turn and keep oscillating with a constant amplitude."""
    trajectory = _run_with_solver(
        PhosphorylationOscillatorSystem(),
        until=40.0,
        dt=0.05,
        solver=SYSTEMS_PRECISE,
    )
    times = trajectory.times
    for name in ('A', 'B', 'C'):
        late = trajectory.values(f'loop.{name}')[times >= 20.0]
        assert late.max() > 0.9
        assert late.min() < 0.1


def test_biochemical_systems_theory_reaches_log_linear_steady_state():
    """Test that both forms reach the steady state that solves `(g - h) log x = log(beta / alpha)`."""
    g = np.array([[0.0, 0.0, -0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
    h = 0.5 * np.eye(3)
    expected = np.exp(np.linalg.solve(g - h, np.log(np.array([1.0, 1.0, 1.0]) / np.array([2.0, 1.0, 1.0]))))
    for form in ('s-system', 'gma'):
        trajectory = _run_with_solver(
            BiochemicalSystemsTheorySystem(form=form),
            until=60.0,
            dt=0.5,
            solver=SYSTEMS_PRECISE,
        )
        final = np.array([trajectory.values(f'X{index}.x')[-1] for index in (1, 2, 3)])
        assert np.allclose(final, expected, rtol=1e-4)


def test_glycolysis_oscillates_and_balances_its_nicotinamide_pool():
    """Test Nielsen et al.: NADH oscillates with a steady period, and `NAD + NADH` relaxes to the feed of 4.24 mM."""
    trajectory = _run_with_solver(GlycolysisOscillationsSystem(), until=600.0, dt=0.5, solver=SYSTEMS_PRECISE)
    nadh = trajectory.values('extract.NADH')[600:]
    peaks = np.flatnonzero((nadh[1:-1] > nadh[:-2]) & (nadh[1:-1] > nadh[2:]))
    periods = np.diff(trajectory.times[600:][1:-1][peaks])
    assert np.ptp(nadh) > 0.1
    assert periods.std() < 0.05 * periods.mean()
    pool = trajectory.values('extract.NAD') + trajectory.values('extract.NADH')
    assert abs(pool[-1] - 4.24) < abs(pool[0] - 4.24)
