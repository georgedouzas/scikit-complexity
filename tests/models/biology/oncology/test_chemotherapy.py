"""Test the `_chemotherapy` module of `skcomplex.models.biology.oncology`."""

import numpy as np

from skcomplex.core.system import System
from skcomplex.models.biology.oncology import (
    ResistanceSwitchingElement,
    de_pillis_radunskaya,
    delayed_metastasis,
    delayed_resting_hunting_chemotherapy,
    drug_resistance_switching,
    multidrug_resistance,
)

from .conftest import _run


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
    assert abs(untreated.values('patient.A')[-1] / A - 1) < 1e-3
    assert abs(untreated.values('patient.N')[-1] / N - 1) < 1e-3
    assert abs(untreated.values('patient.E')[-1] / E - 1) < 1e-3
    treated = _run(de_pillis_radunskaya(u=1.0), until=150.0, dt=0.5)
    assert treated.values('patient.A')[-1] < 0.01 * 0.25
    assert treated.values('patient.N').min() > 0.7


def test_multidrug_resistance_untreated_total_grows_exponentially():
    """Test that mutation conserves the cells, so the untreated total grows as `S0 exp(r t)`."""
    trajectory = _run(multidrug_resistance(u1=0.0, u2=0.0, u3=0.0), until=300.0, dt=10.0)
    names = ('S', 'R1', 'R2', 'R3', 'R12', 'R13', 'R23', 'R123')
    total = sum(trajectory.values(f'tumour.{name}') for name in names)
    assert np.allclose(total, 4.605e11 * np.exp(0.0099 * trajectory.times), rtol=1e-6)
    treated = _run(multidrug_resistance(), until=300.0, dt=10.0)
    assert treated.values('tumour.R123')[-1] > 1e3 * sum(treated.values(f'tumour.{name}')[-1] for name in names[:-1])


def test_drug_resistance_switching_induction_speeds_resistance():
    """Test that resistance induced by the drug lets the resistant cells fill half the capacity sooner."""
    times = {}
    for alpha_m in (0.0, 0.01):
        trajectory = _run(drug_resistance_switching(alpha_m=alpha_m), until=400.0, dt=1.0)
        times[alpha_m] = trajectory.times[np.argmax(trajectory.values('tumour.Rm') > 0.5)]
        assert trajectory.values('tumour.S')[-1] < 1e-6
    assert times[0.01] < 0.6 * times[0.0]


def test_drug_resistance_switching_resensitizes_after_the_dose():
    """Test that switched cells return to sensitivity once the dose stops, so the sensitive cells take back over."""
    switching = {'m1': 0.0, 'alpha_m': 0.0, 'alpha_p': 0.01, 'm5': 0.05}
    treated = _run(drug_resistance_switching(**switching), until=200.0, dt=1.0)
    assert treated.values('tumour.Rp')[-1] > 0.5
    tumour = ResistanceSwitchingElement(
        'tumour',
        S=treated.values('tumour.S')[-1],
        Rp=treated.values('tumour.Rp')[-1],
        u=0.0,
        **switching,
    )
    after = _run(System(components=[tumour]), until=400.0, dt=1.0)
    assert after.values('tumour.S')[-1] > 0.9


def test_delayed_resting_hunting_chemotherapy_infusion_clears_oscillating_tumour():
    """Test that the delay makes the untreated tumour oscillate, and that a constant infusion clears it."""
    untreated = _run(delayed_resting_hunting_chemotherapy(), until=1500.0, dt=5.0)
    late = untreated.values('patient.A')[untreated.times > 500.0]
    assert late.max() > 0.15
    assert late.min() < 0.01
    treated = _run(delayed_resting_hunting_chemotherapy(u=0.03), until=1500.0, dt=5.0)
    assert abs(treated.values('patient.A')[-1]) < 1e-6


def test_delayed_metastasis_tumour_takes_over_both_sites():
    """Test that untreated tumours outcompete the normal cells and settle at their logistic capacities at both sites."""
    trajectory = _run(delayed_metastasis(), until=10.0, dt=0.1)
    assert abs(trajectory.values('patient.Ap')[-1] / ((1 - 0.1 / 10.0) / 0.00047) - 1) < 1e-4
    assert abs(trajectory.values('patient.As')[-1] * 0.00043 - 1) < 1e-3
    assert trajectory.values('patient.Np')[-1] < 1e-3
    assert trajectory.values('patient.Ns')[-1] < 1e-3
    secondary = trajectory.values('patient.As')
    assert secondary[trajectory.times <= 1.0].max() < 10.0
