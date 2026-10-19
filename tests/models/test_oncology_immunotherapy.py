"""Test the immunotherapy models against oscillations, clearance, equilibria, and delays."""

import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.models.biology.oncology import (
    anti_pd1_checkpoint,
    delayed_dc_vaccine,
    dendritic_cell_vaccine,
    kirschner_panetta,
    tumour_immune_suppression,
)

STIFF = ScipyODE(method='LSODA', rtol=1e-8, atol=1e-10)


def _run(system, until, dt):
    """Return the trajectory of a system integrated with a stiff method."""
    return Simulation(system, dt=dt, solver=STIFF).run(until=until).trajectory_


def _count_peaks(values):
    """Return the number of strict local maxima of a series."""
    return int(np.sum((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])))


def test_kirschner_panetta_oscillates_untreated_and_clears_when_treated():
    """Test that an oscillatory antigenicity gives tumour-effector cycles, and adoptive therapy with IL-2 clears it."""
    untreated = _run(kirschner_panetta(alpha_a=0.02), until=1500.0, dt=1.0)
    late = untreated.times >= 500.0
    tumour, effector = untreated['site.A'][late], untreated['site.E'][late]
    assert _count_peaks(tumour) >= 2
    assert _count_peaks(effector) >= 2
    assert tumour.max() > 1e3 * tumour.min()
    assert effector.max() > 10 * effector.min()
    treated = _run(kirschner_panetta(alpha_a=0.02, u1=1.1, u2=1.1), until=1500.0, dt=1.0)
    assert np.all(treated['site.A'][treated.times >= 500.0] < 1e-3)


def test_dendritic_cell_vaccine_shrinks_the_tumour_below_its_equilibrium():
    """Test that the untreated tumour settles at `1 - c1 s2 / (d2 r3)` and that the vaccine shrinks it far below."""
    untreated = _run(dendritic_cell_vaccine(dose=0.0), until=2880.0, dt=24.0)
    assert np.isclose(untreated['site.A'][-1], 1 - 0.1 * 0.02 / 0.02, rtol=1e-6)
    vaccinated = _run(dendritic_cell_vaccine(), until=1344.0, dt=24.0)
    assert vaccinated['site.A'].min() < 0.05


def test_anti_pd1_checkpoint_frees_t_cells_with_the_dose():
    """Test that larger anti-PD-1 infusions leave more T cells and a smaller tumour."""
    runs = [_run(anti_pd1_checkpoint(u_ap=dose), until=150.0, dt=1.0) for dose in (0.0, 1e-10, 1e-9)]
    t_cells = [run['site.E'][-1] for run in runs]
    tumours = [run['site.A'][-1] for run in runs]
    assert t_cells[0] < t_cells[1] < t_cells[2]
    assert tumours[0] > tumours[1] > tumours[2]


def test_delayed_dc_vaccine_helper_response_lags_by_two_divisions():
    """Test that type 1 helper cells ignore the vaccine for one division and respond strongly after two."""
    untreated = _run(delayed_dc_vaccine(dose=0.0), until=1.5, dt=0.05)
    vaccinated = _run(delayed_dc_vaccine(), until=1.5, dt=0.05)
    gain = vaccinated['site.E1H'] - untreated['site.E1H']
    times = untreated.times
    assert np.all(np.abs(gain[times < 0.47]) < 1e-6)
    assert gain[np.isclose(times, 0.9)][0] < 10.0
    assert gain[np.isclose(times, 1.5)][0] > 1e4


def test_tumour_immune_suppression_escapes_only_when_weakly_antigenic():
    """Test that a weakly antigenic tumour escapes to a large size and a strongly antigenic one is held below a cell."""
    weak = _run(tumour_immune_suppression(alpha_a=1e-5), until=200.0, dt=1.0)
    strong = _run(tumour_immune_suppression(alpha_a=100.0), until=200.0, dt=1.0)
    assert weak['site.A'][-1] > 1e8
    assert strong['site.A'][-1] < 1.0
