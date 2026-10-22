"""Test the chemotherapy models against their equilibria, thresholds, and the behaviour the book reports."""

import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.core.system import System
from skcomplex.models.biology.oncology import (
    ResistanceSwitching,
    cell_cycle_chemotherapy,
    de_pillis_radunskaya,
    delayed_metastasis,
    delayed_resting_hunting_chemotherapy,
    drug_resistance_switching,
    erythropoiesis_mercaptopurine,
    multidrug_resistance,
)

PRECISE = ScipyODE(rtol=1e-9, atol=1e-12)


def _run(system, until, dt):
    """Return the trajectory of a system integrated tightly."""
    return Simulation(system, dt=dt, solver=PRECISE).run(until=until).trajectory_


def _high_equilibrium():
    """Return the immune, tumour, and normal cells at the high tumour equilibrium of de Pillis and Radunskaya."""
    s, rho, alpha, c1, c2, c3, c4, d1, r1, r2 = 0.33, 0.01, 0.3, 1.0, 0.5, 1.0, 1.0, 0.2, 1.5, 1.0
    A = np.linspace(0.3, 0.99, 100_000)
    N = 1 - c4 * A / r2
    E = s * (alpha + A) / ((c1 * A + d1) * (alpha + A) - rho * A)
    residual = r1 * (1 - A) - c2 * E - c3 * N
    i = np.argmin(np.abs(residual))
    return E[i], A[i], N[i]


def test_de_pillis_radunskaya_infusion_clears_the_tumour():
    """Test that the untreated tumour settles at its high equilibrium and that a constant infusion clears it."""
    untreated = _run(de_pillis_radunskaya(), until=300.0, dt=1.0)
    E, A, N = _high_equilibrium()
    assert abs(untreated['patient.A'][-1] / A - 1) < 1e-3
    assert abs(untreated['patient.N'][-1] / N - 1) < 1e-3
    assert abs(untreated['patient.E'][-1] / E - 1) < 1e-3
    treated = _run(de_pillis_radunskaya(u=1.0), until=150.0, dt=0.5)
    assert treated['patient.A'][-1] < 0.01 * 0.25
    assert treated['patient.N'].min() > 0.7


def test_multidrug_resistance_untreated_total_grows_exponentially():
    """Test that mutation conserves the cells, so the untreated total grows as `S0 exp(r t)`."""
    trajectory = _run(multidrug_resistance(u1=0.0, u2=0.0, u3=0.0), until=300.0, dt=10.0)
    names = ('S', 'R1', 'R2', 'R3', 'R12', 'R13', 'R23', 'R123')
    total = sum(trajectory[f'tumour.{name}'] for name in names)
    assert np.allclose(total, 4.605e11 * np.exp(0.0099 * trajectory.times), rtol=1e-6)
    treated = _run(multidrug_resistance(), until=300.0, dt=10.0)
    assert treated['tumour.R123'][-1] > 1e3 * sum(treated[f'tumour.{name}'][-1] for name in names[:-1])


def test_drug_resistance_switching_induction_speeds_resistance():
    """Test that resistance induced by the drug lets the resistant cells fill half the capacity sooner."""
    times = {}
    for alpha_m in (0.0, 0.01):
        trajectory = _run(drug_resistance_switching(alpha_m=alpha_m), until=400.0, dt=1.0)
        times[alpha_m] = trajectory.times[np.argmax(trajectory['tumour.Rm'] > 0.5)]
        assert trajectory['tumour.S'][-1] < 1e-6
    assert times[0.01] < 0.6 * times[0.0]


def test_drug_resistance_switching_resensitizes_after_the_dose():
    """Test that switched cells return to sensitivity once the dose stops, so the sensitive cells take back over."""
    switching = {'m1': 0.0, 'alpha_m': 0.0, 'alpha_p': 0.01, 'm5': 0.05}
    treated = _run(drug_resistance_switching(**switching), until=200.0, dt=1.0)
    assert treated['tumour.Rp'][-1] > 0.5
    tumour = ResistanceSwitching('tumour', S=treated['tumour.S'][-1], Rp=treated['tumour.Rp'][-1], u=0.0, **switching)
    after = _run(System(components=[tumour]), until=400.0, dt=1.0)
    assert after['tumour.S'][-1] > 0.9


def test_delayed_resting_hunting_chemotherapy_infusion_clears_oscillating_tumour():
    """Test that the delay makes the untreated tumour oscillate, and that a constant infusion clears it."""
    untreated = _run(delayed_resting_hunting_chemotherapy(), until=1500.0, dt=5.0)
    late = untreated['patient.A'][untreated.times > 500.0]
    assert late.max() > 0.15
    assert late.min() < 0.01
    treated = _run(delayed_resting_hunting_chemotherapy(u=0.03), until=1500.0, dt=5.0)
    assert abs(treated['patient.A'][-1]) < 1e-6


def test_delayed_metastasis_tumour_takes_over_both_sites():
    """Test that untreated tumours outcompete the normal cells and settle at their logistic capacities at both sites."""
    trajectory = _run(delayed_metastasis(), until=10.0, dt=0.1)
    assert abs(trajectory['patient.Ap'][-1] / ((1 - 0.1 / 10.0) / 0.00047) - 1) < 1e-4
    assert abs(trajectory['patient.As'][-1] * 0.00043 - 1) < 1e-3
    assert trajectory['patient.Np'][-1] < 1e-3
    assert trajectory['patient.Ns'][-1] < 1e-3
    secondary = trajectory['patient.As']
    assert secondary[trajectory.times <= 1.0].max() < 10.0


def _growth_rate(trajectory, names):
    """Return the late exponential growth rate of the total cells."""
    total = np.log(sum(trajectory[f'tumour.{name}'] for name in names))
    late = trajectory.times >= 300.0
    return np.polyfit(trajectory.times[late], total[late], 1)[0]


def test_cell_cycle_chemotherapy_kills_above_half():
    """Test that the tumour grows at the dominant eigenvalue of its cycle, which changes sign at a kill of one half."""
    l1, l2, l3 = 0.197, 0.395, 0.107
    for u1, u2 in ((0.4, 0.5), (0.6, 0.0)):
        matrix = np.array(
            [[-l1, 0.0, 2 * (1 - u1) * l3], [l1, -(1 - u2) * l2, 0.0], [0.0, (1 - u2) * l2, -l3]],
        )
        trajectory = _run(cell_cycle_chemotherapy(u1=u1, u2=u2), until=400.0, dt=2.0)
        rate = _growth_rate(trajectory, ('A1', 'A2', 'A3'))
        assert abs(rate - np.linalg.eigvals(matrix).real.max()) < 1e-6
        assert (rate > 0) == (u1 < 0.5)
    dormant = _run(cell_cycle_chemotherapy(dormant=True, u1=0.55, v=1.0), until=400.0, dt=2.0)
    assert _growth_rate(dormant, ('A0', 'A1', 'A2')) < 0


def test_erythropoiesis_mercaptopurine_steady_volume():
    """Test that the mean volume settles at `MCV0 + g1 U` and the mature cells at `r1 Q / d1`."""
    trajectory = _run(erythropoiesis_mercaptopurine(U=500.0), until=6000.0, dt=20.0)
    assert abs(trajectory['patient.MCV'][-1] / (85.0 + 0.0317 * 500.0) - 1) < 1e-3
    assert abs(trajectory['patient.L'][-1] / (0.00139 * trajectory['patient.Q'][-1] / 0.0071) - 1) < 1e-3
    assert trajectory['patient.L'][-1] < 0.9 * trajectory['patient.L'][0]
