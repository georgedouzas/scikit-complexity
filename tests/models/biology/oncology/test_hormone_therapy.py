"""Test the `_hormone_therapy` module of `skcomplex.models.biology.oncology`."""

import numpy as np

from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.models.biology.oncology import (
    continuous_androgen_suppression,
    ideta_intermittent_androgen,
    piecewise_linear_ias,
    prostate_androgen_receptor,
    prostate_cell_quota,
    tamoxifen_cell_cycle,
)

from ..conftest import _run_seeded

STIFF = ScipyODESolver(method='LSODA', rtol=1e-10, atol=1e-12)


def test_continuous_androgen_suppression_relapses_through_independent_cells():
    """Test that the androgen falls to `U0 (1 - a)` in closed form and that independent cells replace dependent ones."""
    trajectory = _run_seeded(continuous_androgen_suppression(), until=1500.0, dt=5.0)
    times = trajectory.times
    exact = 20.0 * (1 - 0.9) + 20.0 * 0.9 * np.exp(-0.08 * times)
    assert np.allclose(trajectory['tumour.U'], exact, rtol=1e-6)
    dependent, independent = trajectory['tumour.AD'], trajectory['tumour.AI']
    assert np.all(np.diff(dependent[times >= 100.0]) < 0)
    assert independent[-1] > 1e3 * dependent[-1]


def test_ideta_intermittent_androgen_switches_at_thresholds():
    """Test that the therapy starts above the upper PSA threshold, stops below the lower one, and the PSA cycles."""
    h0, h1 = 10.0, 15.0
    trajectory = _run_seeded(ideta_intermittent_androgen(h0=h0, h1=h1), until=600.0, dt=1.0)
    names = list(trajectory.problem.states.names)
    psa = trajectory['tumour.AD'] + trajectory['tumour.AI']
    labels = [event.label for event in trajectory.events]
    assert len(labels) >= 8
    assert labels == ['on', 'off'] * (len(labels) // 2) + ['on'] * (len(labels) % 2)
    for event in trajectory.events:
        before = event.before[names.index('tumour.AD')] + event.before[names.index('tumour.AI')]
        assert abs(before - (h1 if event.label == 'on' else h0)) < 1e-6
        assert event.after[names.index('tumour.u')] == (1.0 if event.label == 'on' else 0.0)
    times = trajectory.times
    first_on = trajectory.events[0].time
    cycling = psa[times > first_on]
    assert cycling.min() > h0 - 1.0
    assert cycling.max() < h1 + 3.0
    for on, off in zip(trajectory.events[::2], trajectory.events[1::2], strict=False):
        assert np.all(trajectory['tumour.u'][(times > on.time) & (times < off.time)] == 1.0)


def test_prostate_cell_quota_reaches_the_quota_equilibrium():
    """Test that without therapy each quota settles at the equilibrium of its linear balance."""
    trajectory = _run_seeded(prostate_cell_quota(width=0.0), until=300.0, dt=5.0)
    uptake = 0.275 * 15.0 / (15.0 + 4.0)
    for name, q in (('tumour.Q1', 0.3), ('tumour.Q2', 0.15)):
        equilibrium = (uptake * 5.0 / (5.0 - q) + 0.025 * q) / (uptake / (5.0 - q) + 0.025 + 0.09)
        assert abs(trajectory[name][-1] / equilibrium - 1) < 1e-6


def test_prostate_androgen_receptor_normal_cells_fill_their_capacity():
    """Test that the normal epithelium without cancer settles at `b1 (1 - d1 / r1)` for the prostate volume."""
    UT = UD = 40.0
    trajectory = _run_seeded(
        prostate_androgen_receptor(AD=0.0, AI=0.0, age=100.0),
        until=1.5e5,
        dt=5000.0,
        solver=STIFF,
    )
    age = 100.0 + 1.5e5 / 8760
    capacity = (18.397 * age**12 / (63.753**12 + age**12) + 28.4) / 5.56
    growth = 1.202e-4 * UD**2 / (12.355**2 + UD**2) + 1.281e-4 * UT**2 / (49.587**2 + UT**2)
    death = 0.008 / (62.162 + UD + UT)
    assert abs(trajectory['prostate.N'][-1] / (capacity * (1 - death / growth)) - 1) < 1e-3


def test_piecewise_linear_ias_decays_exponentially_on_therapy():
    """Test that the dependent cells decay as `exp(w11 t)` during the first therapy period and that therapy cycles."""
    trajectory = _run_seeded(piecewise_linear_ias(patient=1), until=1500.0, dt=1.0)
    times = trajectory.times
    first_off = trajectory.events[0]
    assert first_off.label == 'off'
    on = times <= first_off.time
    assert np.allclose(trajectory['tumour.AD'][on], 17.84 * np.exp((0.956156 - 1) * times[on]), rtol=1e-6)
    assert len(trajectory.events) > 20


def test_tamoxifen_cell_cycle_keeps_lymphocytes_at_their_influx_balance():
    """Test that without the drug the lymphocytes settle at `s / d1` and that tamoxifen shrinks the tumour."""
    untreated = _run_seeded(tamoxifen_cell_cycle(u=0.0), until=2400.0, dt=24.0, solver=STIFF)
    assert abs(untreated['tumour.E'][-1] / (0.121 / 0.012) - 1) < 1e-6
    treated = _run_seeded(tamoxifen_cell_cycle(), until=2400.0, dt=24.0, solver=STIFF)
    volume = treated['tumour.Ag'] + treated['tumour.As'] + treated['tumour.Am']
    assert volume[-1] < 0.2 * volume[0]
    assert treated['tumour.E'][-1] < untreated['tumour.E'][-1]
