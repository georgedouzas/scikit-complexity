"""Test the `_stem_cells` module of `skcomplex.models.biology.oncology`."""

from skcomplex.engine.solvers import ScipyODESolver
from skcomplex.models.biology.oncology import (
    bone_marrow_blood_leukaemia,
    leukaemia_stem_cells,
    stem_cell_differentiation,
)

from .conftest import STIFF, _run_optional_solver


def test_stem_cell_differentiation_settles_under_continuous_therapy():
    """Test that the seed fixes the mutation rate and that the lineages settle with a malignant population."""
    system = stem_cell_differentiation(seed=3)
    m1 = system.elements[0].m1
    assert 0.005 <= m1 < 0.01
    assert stem_cell_differentiation(seed=3).elements[0].m1 == m1
    trajectory = _run_optional_solver(system, until=6000.0, dt=10.0, solver=STIFF)
    for state in ('NSq', 'NPr', 'ND', 'APr', 'AD'):
        values = trajectory[f'tissue.{state}']
        assert values[-1] > 0
        assert abs(values[-1] / values[-21] - 1) < 1e-3


def test_bone_marrow_blood_leukaemia_blasts_crowd_out_normal_cells():
    """Test that the blasts take over the marrow and that the normal blood cells fall by orders of magnitude."""
    trajectory = _run_optional_solver(
        bone_marrow_blood_leukaemia(),
        until=2000.0,
        dt=10.0,
        solver=ScipyODESolver(method='BDF'),
    )
    assert trajectory['patient.ABM'][-1] > 10 * trajectory['patient.NBM'][-1]
    assert trajectory['patient.NPB'][-1] < 1e-3 * trajectory['patient.NPB'][0]


def test_leukaemia_stem_cells_exclude_the_weaker_stem_cells():
    """Test that the stem cells with the larger net renewal exclude the others from the niche."""
    controlled = _run_optional_solver(leukaemia_stem_cells(r4=0.5), until=500.0, dt=5.0)
    healthy = _run_optional_solver(leukaemia_stem_cells(AS=0.0), until=500.0, dt=5.0)
    assert controlled['niche.AS'][-1] < 1e-3
    assert abs(controlled['niche.NS'][-1] / healthy['niche.NS'][-1] - 1) < 1e-3
    leukaemic = _run_optional_solver(leukaemia_stem_cells(), until=500.0, dt=5.0)
    assert leukaemic['niche.NS'][-1] < 0.02 * leukaemic['niche.AS'][-1]
