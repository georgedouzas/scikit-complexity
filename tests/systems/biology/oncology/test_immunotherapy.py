"""Test the `_immunotherapy` module of `skcomplex.systems.biology.oncology`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.systems.biology.oncology import (
    AntiPD1CheckpointSystem,
    DelayedDCVaccineSystem,
    DendriticCellVaccineSystem,
    KirschnerPanettaSystem,
    KirschnerPanettaTumourElement,
    TumourImmuneSuppressionSystem,
)

STIFF = ScipyODESolver(method='LSODA', rtol=1e-8, atol=1e-10)


def _run(system, until, dt):
    """Return the trajectory of a system integrated with a stiff method."""
    return simulate(system, dt=dt, solver=STIFF, until=until)


def _count_peaks(values):
    """Return the number of strict local maxima of a series."""
    return int(np.sum((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])))


def test_kirschner_panetta_oscillates_untreated_and_clears_when_treated():
    """Test that an oscillatory antigenicity gives tumour-effector cycles, and adoptive therapy with IL-2 clears it."""
    untreated = _run(KirschnerPanettaSystem(alpha_a=0.02), until=1500.0, dt=1.0)
    late = untreated.times >= 500.0
    tumour, effector = untreated.get_values('site.A')[late], untreated.get_values('site.E')[late]
    assert _count_peaks(tumour) >= 2
    assert _count_peaks(effector) >= 2
    assert tumour.max() > 1e3 * tumour.min()
    assert effector.max() > 10 * effector.min()
    treated = _run(KirschnerPanettaSystem(alpha_a=0.02, u1=1.1, u2=1.1), until=1500.0, dt=1.0)
    assert np.all(treated.get_values('site.A')[treated.times >= 500.0] < 1e-3)


def test_dendritic_cell_vaccine_shrinks_the_tumour_below_its_equilibrium():
    """Test that the untreated tumour settles at `1 - c1 s2 / (d2 r3)` and that the vaccine shrinks it far below."""
    untreated = _run(DendriticCellVaccineSystem(dose=0.0), until=2880.0, dt=24.0)
    assert np.isclose(untreated.get_values('site.A')[-1], 1 - 0.1 * 0.02 / 0.02, rtol=1e-6)
    vaccinated = _run(DendriticCellVaccineSystem(), until=1344.0, dt=24.0)
    assert vaccinated.get_values('site.A').min() < 0.05


def test_anti_pd1_checkpoint_frees_t_cells_with_the_dose():
    """Test that larger anti-PD-1 infusions leave more T cells and a smaller tumour."""
    runs = [_run(AntiPD1CheckpointSystem(u_ap=dose), until=150.0, dt=1.0) for dose in (0.0, 1e-10, 1e-9)]
    t_cells = [run.get_values('site.E')[-1] for run in runs]
    tumours = [run.get_values('site.A')[-1] for run in runs]
    assert t_cells[0] < t_cells[1] < t_cells[2]
    assert tumours[0] > tumours[1] > tumours[2]


def test_delayed_dc_vaccine_helper_response_lags_by_two_divisions():
    """Test that type 1 helper cells ignore the vaccine for one division and respond strongly after two."""
    untreated = _run(DelayedDCVaccineSystem(dose=0.0), until=1.5, dt=0.05)
    vaccinated = _run(DelayedDCVaccineSystem(), until=1.5, dt=0.05)
    gain = vaccinated.get_values('site.E1H') - untreated.get_values('site.E1H')
    times = untreated.times
    assert np.all(np.abs(gain[times < 0.47]) < 1e-6)
    assert gain[np.isclose(times, 0.9)][0] < 10.0
    assert gain[np.isclose(times, 1.5)][0] > 1e4


def test_tumour_immune_suppression_escapes_only_when_weakly_antigenic():
    """Test that a weakly antigenic tumour escapes to a large size and a strongly antigenic one is held below a cell."""
    weak = _run(TumourImmuneSuppressionSystem(alpha_a=1e-5), until=200.0, dt=1.0)
    strong = _run(TumourImmuneSuppressionSystem(alpha_a=100.0), until=200.0, dt=1.0)
    assert weak.get_values('site.A')[-1] > 1e8
    assert strong.get_values('site.A')[-1] < 1.0


def test_kirschner_panetta_is_not_registered_twice():
    """Test that the scaled tumour of the immunotherapy system is a new element, not the book's dimensional one."""
    assert KirschnerPanettaTumourElement('site').alpha1 == 2e7
