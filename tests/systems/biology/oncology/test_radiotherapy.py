"""Test the `_radiotherapy` module of `skcomplex.systems.biology.oncology`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.systems.biology.oncology import LQRadiosensitizerSystem, RadiotherapyCompetitionSystem

PRECISE = ScipyODESolver(rtol=1e-10, atol=1e-12)


def _run(system, until, dt):
    """Return the trajectory of a system integrated tightly."""
    return simulate(system, dt=dt, random_state=0, solver=PRECISE, until=until)


def test_lq_radiosensitizer_fraction_survival():
    """Test that a fraction of dose `d` without radiosensitizer leaves `exp(-alpha d - beta d^2)` of the cells."""
    alpha, beta, dose = 0.054, 0.0054, 2.0
    trajectory = _run(LQRadiosensitizerSystem(Us=0.0, alpha=alpha, beta=beta, dose=dose), until=15.0, dt=0.5)
    index = list(trajectory.problem.states.names).index('tumour.A1')
    fractions = [event for event in trajectory.events if event.label == 'fraction']
    assert len(fractions) == 10
    for event in fractions:
        assert abs(event.after[index] / event.before[index] - np.exp(-alpha * dose - beta * dose**2)) < 1e-6


def test_lq_radiosensitizer_regrows_at_the_inhibited_rate():
    """Test that after the last fraction the cells regrow at `r exp(-a_gamma u_tot)` with `u_tot = n d`."""
    trajectory = _run(LQRadiosensitizerSystem(Us=0.0), until=60.0, dt=0.5)
    times, cells = trajectory.times, trajectory.get_values('tumour.A1')
    assert np.allclose(trajectory.get_values('tumour.u_tot')[-1], 10 * 2.0)
    after = times >= 10.0
    rate = 0.04 * np.exp(-0.004 * 20.0)
    exact = cells[after][0] * np.exp(rate * (times[after] - times[after][0]))
    assert np.allclose(cells[after], exact, rtol=1e-6)


def test_radiotherapy_competition_lets_normal_cells_survive():
    """Test that the tumour excludes the normal cells without radiation and that irradiation lets them coexist."""
    untreated = simulate(RadiotherapyCompetitionSystem(a2=0.0, epsilon=0.0), dt=20.0, until=3000.0)
    assert untreated.get_values('tissue.N')[-1] < 1e-3
    assert abs(untreated.get_values('tissue.A')[-1] - 1.0) < 1e-3
    treated = simulate(RadiotherapyCompetitionSystem(), dt=20.0, until=2000.0)
    assert treated.get_values('tissue.N')[-1] > 0.05
    assert treated.get_values('tissue.A')[-1] < 0.8
