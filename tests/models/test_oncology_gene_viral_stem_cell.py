"""Test the gene, viral, and stem-cell therapy models against their equilibria and their known behaviours."""

import numpy as np
from scipy.optimize import fsolve

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.models.biology.oncology import (
    bone_marrow_blood_leukaemia,
    leukaemia_stem_cells,
    oncolytic_virus,
    oncolytic_virus_dendritic,
    sirna_tgf_beta,
    stem_cell_differentiation,
    tcr_gene_therapy,
)

STIFF = ScipyODE(method='LSODA', rtol=1e-8, atol=1e-10)


def _run(system, until, dt, solver=None):
    """Return the trajectory of a system."""
    settings = {} if solver is None else {'solver': solver}
    return Simulation(system, dt=dt, random_state=0, **settings).run(until=until).trajectory_


def _tcr_rates(y):
    """Return the rates of the gene therapy model with the book's parameters."""
    E, A = y
    return [1 + 0.05 * A - 0.03 * E + 0.1245 * E / (E + 1e-3), 0.18 * A * (1 - 1e-9 * A) - E * A / (A + 1e5)]


def test_tcr_gene_therapy_settles_on_a_dormant_tumour():
    """Test that the tumour and effector cells settle on the coexistence equilibrium far below the capacity."""
    trajectory = _run(tcr_gene_therapy(), until=1500.0, dt=5.0)
    E, A = trajectory['site.E'], trajectory['site.A']
    equilibrium = fsolve(_tcr_rates, [E[-1], A[-1]], xtol=1e-12)
    assert np.allclose([E[-1], A[-1]], equilibrium, rtol=1e-2)
    assert A.max() < 1e-3 / 1e-9
    assert np.any(np.diff(np.sign(np.diff(A))) != 0)


def test_sirna_tgf_beta_lets_the_tumour_escape_and_sirna_clears_it():
    """Test that the immune system holds the tumour without TGF-beta, TGF-beta lets it escape, and siRNA clears it."""
    without = _run(sirna_tgf_beta(rho4=0.0), until=200.0, dt=1.0, solver=STIFF)['site.A']
    escaped = _run(sirna_tgf_beta(), until=200.0, dt=1.0, solver=STIFF)['site.A']
    treated = _run(sirna_tgf_beta(u=1e8), until=200.0, dt=1.0, solver=STIFF)['site.A']
    assert without.max() < 1e5
    assert escaped[-1] > 1e9
    assert treated[-1] < 1.0


def test_oncolytic_virus_settles_on_the_coexistence_point():
    """Test that the virus reduces the tumour to the coexistence point of the infected and uninfected cells."""
    alpha, r, d = 1.2, 0.8, 0.6
    trajectory = _run(oncolytic_virus(alpha=alpha, r=r, d=d), until=200.0, dt=1.0)
    infected = (1 - d / alpha) / (1 + alpha - r)
    uninfected = d / alpha - r * infected
    assert abs(trajectory['tumour.AV'][-1] / infected - 1) < 1e-4
    assert abs(trajectory['tumour.AU'][-1] / uninfected - 1) < 1e-4


def test_oncolytic_virus_dendritic_clears_the_tumour_with_the_virus():
    """Test that the untreated tumour grows exponentially and that the virus alone clears it."""
    untreated = _run(oncolytic_virus_dendritic(uv=0.0, ud=0.0), until=20.0, dt=1.0, solver=STIFF)
    assert np.allclose(untreated['site.AU'], 60.0 * np.exp(0.3198 * untreated.times), rtol=1e-5)
    virus = _run(oncolytic_virus_dendritic(ud=0.0), until=20.0, dt=1.0, solver=STIFF)
    assert virus['site.AU'][-1] + virus['site.AV'][-1] < 1e-6


def test_stem_cell_differentiation_settles_under_continuous_therapy():
    """Test that the seed fixes the mutation rate and that the lineages settle with a malignant population."""
    system = stem_cell_differentiation(seed=3)
    m1 = system.elements[0].m1
    assert 0.005 <= m1 < 0.01
    assert stem_cell_differentiation(seed=3).elements[0].m1 == m1
    trajectory = _run(system, until=6000.0, dt=10.0, solver=STIFF)
    for state in ('NSq', 'NPr', 'ND', 'APr', 'AD'):
        values = trajectory[f'tissue.{state}']
        assert values[-1] > 0
        assert abs(values[-1] / values[-21] - 1) < 1e-3


def test_bone_marrow_blood_leukaemia_blasts_crowd_out_normal_cells():
    """Test that the blasts take over the marrow and that the normal blood cells fall by orders of magnitude."""
    trajectory = _run(bone_marrow_blood_leukaemia(), until=2000.0, dt=10.0, solver=ScipyODE(method='BDF'))
    assert trajectory['patient.ABM'][-1] > 10 * trajectory['patient.NBM'][-1]
    assert trajectory['patient.NPB'][-1] < 1e-3 * trajectory['patient.NPB'][0]


def test_leukaemia_stem_cells_exclude_the_weaker_stem_cells():
    """Test that the stem cells with the larger net renewal exclude the others from the niche."""
    controlled = _run(leukaemia_stem_cells(r4=0.5), until=500.0, dt=5.0)
    healthy = _run(leukaemia_stem_cells(AS=0.0), until=500.0, dt=5.0)
    assert controlled['niche.AS'][-1] < 1e-3
    assert abs(controlled['niche.NS'][-1] / healthy['niche.NS'][-1] - 1) < 1e-3
    leukaemic = _run(leukaemia_stem_cells(), until=500.0, dt=5.0)
    assert leukaemic['niche.NS'][-1] < 0.02 * leukaemic['niche.AS'][-1]
