"""Test the combination therapy models against the behaviour their sources describe."""

import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.models.biology.oncology import (
    angiogenesis_immunotherapy,
    chemoimmunotherapy,
    chemovirotherapy,
    clonal_chemoimmunotherapy,
    glioma_chemovirotherapy,
    her2_targeted_therapy,
)

PRECISE = ScipyODE(method='LSODA', rtol=1e-8, atol=1e-10)
STIFF = ScipyODE(method='LSODA', rtol=1e-6, atol=1e-6)


def _run(system, until, dt, solver='auto'):
    """Return the trajectory of a system."""
    return Simulation(system, dt=dt, random_state=0, solver=solver).run(until=until).trajectory_


def test_chemovirotherapy_combination_beats_each_therapy_alone():
    """Test that the tumour at the end of the run is smallest under the combination of chemotherapy and virus."""
    tumours = {}
    for therapy, settings in {'chemotherapy': {'V': 0.0}, 'virus': {'u1': 0.0}, 'both': {}}.items():
        trajectory = _run(chemovirotherapy(**settings), until=32.0, dt=0.1)
        tumours[therapy] = trajectory['tumour.AU'][-1] + trajectory['tumour.AV'][-1]
    assert tumours['both'] < 0.95 * min(tumours['chemotherapy'], tumours['virus'])


def test_chemovirotherapy_drug_relaxes_to_infusion_balance():
    """Test that a constant infusion drives the drug along `u1 / dU + (U0 - u1 / dU) exp(-dU t)`."""
    trajectory = _run(chemovirotherapy(V=0.0, U=0.1, u1=10.0), until=5.0, dt=0.1, solver=PRECISE)
    exact = 10.0 / 4.16 + (0.1 - 10.0 / 4.16) * np.exp(-4.16 * trajectory.times)
    assert np.allclose(trajectory['tumour.U'], exact, rtol=1e-5)


def test_chemoimmunotherapy_combination_clears_the_tumour():
    """Test that chemotherapy and immunotherapy each fail alone and clear the tumour together."""
    alone = {
        'chemotherapy': {'u_til': 0.0, 'u_il2': 0.0},
        'immunotherapy': {'u_chemo': 0.0},
    }
    for settings in alone.values():
        assert _run(chemoimmunotherapy(**settings), until=60.0, dt=0.5, solver=STIFF)['tumour.A'][-1] > 1e7
    assert _run(chemoimmunotherapy(), until=60.0, dt=0.5, solver=STIFF)['tumour.A'][-1] < 1.0


def test_clonal_chemoimmunotherapy_chemotherapy_selects_the_resistant_clone():
    """Test that chemotherapy alone removes the sensitive clone and lets the resistant clone take over."""
    trajectory = _run(clonal_chemoimmunotherapy(u_vaccine=0.0), until=300.0, dt=1.0)
    assert trajectory['tumour.A1'][-1] < 1.0
    assert trajectory['tumour.A2'][-1] > 100 * trajectory['tumour.A2'][0]
    combined = _run(clonal_chemoimmunotherapy(), until=300.0, dt=1.0)
    assert combined['tumour.A1'][-1] + combined['tumour.A2'][-1] < 1.0


def test_glioma_chemovirotherapy_spares_the_neurons():
    """Test that untreated glioma takes glia and neurons with it, while therapy clears it and spares the neurons."""
    untreated = _run(glioma_chemovirotherapy(V=0.0, u1=0.0), until=300.0, dt=5.0)
    assert untreated['brain.AU'][-1] > 10 * untreated['brain.AU'][0]
    assert untreated['brain.N'][-1] < 0.75 * untreated['brain.N'][0]
    treated = _run(glioma_chemovirotherapy(), until=300.0, dt=5.0)
    assert treated['brain.AU'][-1] < 1.0
    assert treated['brain.N'][-1] > 0.98 * treated['brain.N'][0]


def test_her2_targeted_therapy_antibody_slows_logistic_growth():
    """Test that the bound antibody reaches the free antibody and slows logistic growth to `r (1 - a1 Ab)`."""
    trajectory = _run(her2_targeted_therapy(antibody=0.2, paclitaxel=0.0), until=10.0, dt=0.05, solver=PRECISE)
    times, A = trajectory.times, trajectory['tumour.A']
    assert np.allclose(trajectory['tumour.Ab'][times >= 1.5], 0.2, rtol=1e-6)
    after = times >= 2.0
    rate, start = 0.83 * (1 - 2.51 * 0.2), A[after][0]
    exact = 0.67 / (1 + (0.67 / start - 1) * np.exp(-rate * (times[after] - 2.0)))
    assert np.allclose(A[after], exact, rtol=1e-5)


def test_angiogenesis_immunotherapy_follows_pharmacokinetics_and_resistance():
    """Test that a dose follows the two-compartment closed form and the resistance integrates the low amounts."""
    trajectory = _run(angiogenesis_immunotherapy(doses=1), until=20.0, dt=0.05, solver=PRECISE)
    times, amount = trajectory.times, trajectory['tumour.c_an']
    alpha, beta, k21, dose = 0.4811, 0.0348, 0.215, 15.0
    later = times >= 1.0
    elapsed = times[later] - 1 / 48
    exact = dose * ((alpha - k21) * np.exp(-alpha * elapsed) + (k21 - beta) * np.exp(-beta * elapsed)) / (alpha - beta)
    assert np.allclose(amount[later], exact, rtol=1e-2)
    shortfall = np.maximum(0.0, 5.0 - amount[later])
    integral = np.concatenate([[0.0], np.cumsum((shortfall[1:] + shortfall[:-1]) / 2 * np.diff(times[later]))])
    resistance = trajectory['tumour.R_an'][later]
    assert np.allclose(resistance - resistance[0], integral, atol=1e-3)
    assert integral[-1] > 1.0
