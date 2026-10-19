"""Test the tumour growth laws and the cell kill models against their closed forms."""

import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.models.biology.oncology import (
    dynamic_carrying_capacity,
    exponential_linear_growth,
    generalized_logistic,
    gompertz,
    norton_simon,
    power_law_growth,
    skipper_log_kill,
    von_bertalanffy,
)
from skcomplex.models.biology.physiology import drug_toxicity

PRECISE = ScipyODE(rtol=1e-10, atol=1e-12)


def _run(system, until, dt):
    """Return the trajectory of a system integrated tightly."""
    return Simulation(system, dt=dt, solver=PRECISE).run(until=until).trajectory_


def test_gompertz_matches_closed_form():
    """Test that Gompertz growth from a small tumour follows its closed form and approaches the capacity."""
    trajectory = _run(gompertz(n=10.0, r=0.0719, K=1964.0), until=150.0, dt=1.0)
    exact = 1964.0 * np.exp(np.log(10.0 / 1964.0) * np.exp(-0.0719 * trajectory.times))
    assert np.max(np.abs(trajectory['tumour.n'] / exact - 1)) < 1e-6
    assert abs(trajectory['tumour.n'][-1] / 1964.0 - 1) < 1e-3


def test_exponential_linear_growth_switches_to_linear():
    """Test that the tumour grows exponentially until `r0 n` reaches `r`, at `tau`, and linearly at rate `r` after."""
    trajectory = _run(exponential_linear_growth(n=2.0, r0=0.31, r=67.8), until=40.0, dt=0.5)
    times, n = trajectory.times, trajectory['tumour.n']
    tau = np.log(67.8 / (0.31 * 2.0)) / 0.31
    early = times <= tau
    assert np.allclose(n[early], 2.0 * np.exp(0.31 * times[early]), rtol=1e-6)
    assert np.allclose(n[~early], 67.8 / 0.31 + 67.8 * (times[~early] - tau), rtol=1e-6)


def test_power_law_growth_matches_closed_form():
    """Test that power-law growth follows `(n0^(1-p) + (1-p) r t)^(1/(1-p))`."""
    trajectory = _run(power_law_growth(n=10.0, r=1.32, p=0.58), until=40.0, dt=1.0)
    exact = (10.0**0.42 + 0.42 * 1.32 * trajectory.times) ** (1 / 0.42)
    assert np.allclose(trajectory['tumour.n'], exact, rtol=1e-6)


def test_von_bertalanffy_matches_closed_form():
    """Test that von Bertalanffy growth follows its closed form and tends to `(r / d)^(1/(1-p))`."""
    r, p, d = 2.32, 2 / 3, 0.808
    trajectory = _run(von_bertalanffy(n=1.0, r=r, p=p, d=d), until=40.0, dt=1.0)
    exact = (r / d + (1.0 - r / d) * np.exp(-d * (1 - p) * trajectory.times)) ** (1 / (1 - p))
    assert np.allclose(trajectory['tumour.n'], exact, rtol=1e-6)
    assert abs(trajectory['tumour.n'][-1] / (r / d) ** 3 - 1) < 1e-2


def test_generalized_logistic_matches_closed_form():
    """Test that generalized logistic growth follows its closed form for an exponent other than 1."""
    r, K, p, n0 = 0.305, 1221.0, 2.0, 10.0
    trajectory = _run(generalized_logistic(n=n0, r=r, K=K, p=p), until=40.0, dt=1.0)
    exact = n0 * K / (n0**p + (K**p - n0**p) * np.exp(-r * p * trajectory.times)) ** (1 / p)
    assert np.allclose(trajectory['tumour.n'], exact, rtol=1e-6)


def test_dynamic_carrying_capacity_raises_the_capacity():
    """Test that the capacity grows by `d n^(2/3)` and that the tumour stays close to it once it has filled it."""
    trajectory = _run(dynamic_carrying_capacity(), until=20.0, dt=0.5)
    n, K = trajectory['tumour.n'], trajectory['vessels.K']
    assert np.all(np.diff(K) > 0)
    assert K[-1] > 20 * K[0]
    assert np.all(np.abs(np.log(n[trajectory.times > 5.0] / K[trajectory.times > 5.0])) < 0.1)


def test_skipper_log_kill_lowers_the_logarithm_by_equal_steps():
    """Test that equal doses on a non-growing tumour each lower its logarithm by `a dose width`."""
    trajectory = _run(skipper_log_kill(n=1e6, r=0.0, a=2.0, dose=1.0, period=7.0, width=1.0), until=35.0, dt=1.0)
    starts = np.log(trajectory['tumour.n'][np.isin(trajectory.times, 7.0 * np.arange(6))])
    assert np.allclose(np.diff(starts), -2.0, atol=1e-6)


def test_norton_simon_regrows_faster_after_a_dose():
    """Test that a Gompertzian tumour regrows faster just after a dose than before it, at a smaller size."""
    trajectory = _run(norton_simon(), until=63.0, dt=0.5)
    times, n = trajectory.times, trajectory['tumour.n']
    rate = np.gradient(np.log(n), times)
    before, after = rate[np.isclose(times, 41.5)][0], rate[np.isclose(times, 44.0)][0]
    assert n[np.isclose(times, 43.0)][0] < n[np.isclose(times, 42.0)][0]
    assert after > before


def test_drug_toxicity_integrates_the_concentration():
    """Test that a cumulative toxicity is the integral of the concentration, and that a cleared one decays after."""
    cumulative = _run(drug_toxicity(eta=0.0, k=0.5, dose=1.0), until=84.0, dt=0.2)
    times = cumulative.times
    infused = np.clip(times[:, None] - 7.0 * np.arange(6), 0.0, 1.0).sum(axis=1)
    assert np.allclose(cumulative['patient.T'], (infused - cumulative['patient.c']) / 0.5, atol=1e-6)
    cleared = _run(drug_toxicity(eta=0.05), until=84.0, dt=0.2)
    late = cleared.times >= 60.0
    assert np.all(np.diff(cleared['patient.T'][late]) < 0)
