"""Test the `_gene_viral_therapy` module of `skcomplex.systems.biology.oncology`."""

import numpy as np
from scipy.optimize import fsolve

from skcomplex.systems.biology.oncology import (
    oncolytic_virus,
    oncolytic_virus_dendritic,
    sirna_tgf_beta,
    tcr_gene_therapy,
)

from .conftest import STIFF, _run_optional_solver


def _tcr_rates(y):
    """Return the rates of the gene therapy system with the book's parameters."""
    E, A = y
    return [1 + 0.05 * A - 0.03 * E + 0.1245 * E / (E + 1e-3), 0.18 * A * (1 - 1e-9 * A) - E * A / (A + 1e5)]


def test_tcr_gene_therapy_settles_on_a_dormant_tumour():
    """Test that the tumour and effector cells settle on the coexistence equilibrium far below the capacity."""
    trajectory = _run_optional_solver(tcr_gene_therapy(), until=1500.0, dt=5.0)
    E, A = trajectory.values('site.E'), trajectory.values('site.A')
    equilibrium = fsolve(_tcr_rates, [E[-1], A[-1]], xtol=1e-12)
    assert np.allclose([E[-1], A[-1]], equilibrium, rtol=1e-2)
    assert A.max() < 1e-3 / 1e-9
    assert np.any(np.diff(np.sign(np.diff(A))) != 0)


def test_sirna_tgf_beta_lets_the_tumour_escape_and_sirna_clears_it():
    """Test that the immune system holds the tumour without TGF-beta, TGF-beta lets it escape, and siRNA clears it."""
    without = _run_optional_solver(sirna_tgf_beta(rho4=0.0), until=200.0, dt=1.0, solver=STIFF).values('site.A')
    escaped = _run_optional_solver(sirna_tgf_beta(), until=200.0, dt=1.0, solver=STIFF).values('site.A')
    treated = _run_optional_solver(sirna_tgf_beta(u=1e8), until=200.0, dt=1.0, solver=STIFF).values('site.A')
    assert without.max() < 1e5
    assert escaped[-1] > 1e9
    assert treated[-1] < 1.0


def test_oncolytic_virus_settles_on_the_coexistence_point():
    """Test that the virus reduces the tumour to the coexistence point of the infected and uninfected cells."""
    alpha, r, d = 1.2, 0.8, 0.6
    trajectory = _run_optional_solver(oncolytic_virus(alpha=alpha, r=r, d=d), until=200.0, dt=1.0)
    infected = (1 - d / alpha) / (1 + alpha - r)
    uninfected = d / alpha - r * infected
    assert abs(trajectory.values('tumour.AV')[-1] / infected - 1) < 1e-4
    assert abs(trajectory.values('tumour.AU')[-1] / uninfected - 1) < 1e-4


def test_oncolytic_virus_dendritic_clears_the_tumour_with_the_virus():
    """Test that the untreated tumour grows exponentially and that the virus alone clears it."""
    untreated = _run_optional_solver(
        oncolytic_virus_dendritic(uv=0.0, ud=0.0),
        until=20.0,
        dt=1.0,
        solver=STIFF,
    )
    assert np.allclose(untreated.values('site.AU'), 60.0 * np.exp(0.3198 * untreated.times), rtol=1e-5)
    virus = _run_optional_solver(oncolytic_virus_dendritic(ud=0.0), until=20.0, dt=1.0, solver=STIFF)
    assert virus.values('site.AU')[-1] + virus.values('site.AV')[-1] < 1e-6
