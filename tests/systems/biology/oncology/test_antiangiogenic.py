"""Test the `_antiangiogenic` module of `skcomplex.systems.biology.oncology`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.systems.biology.oncology import HahnfeldtSystem, VascularCorePeripherySystem

STIFF = ScipyODESolver(method='LSODA', rtol=1e-8, atol=1e-8)


def _run(system, until, dt):
    """Return the trajectory of a system integrated with a stiff method."""
    return simulate(system, dt=dt, solver=STIFF, until=until)


def test_hahnfeldt_approaches_the_equilibrium_volume():
    """Test that the untreated tumour and carrying capacity approach `(b/d)^(3/2)` within 1%."""
    trajectory = _run(HahnfeldtSystem(A=200.0, Vb=625.0, dose=0.0), until=100.0, dt=1.0)
    equilibrium = (5.85 / 0.00873) ** 1.5
    assert abs(trajectory.get_values('tumour.A')[-1] / equilibrium - 1) < 1e-2
    assert abs(trajectory.get_values('tumour.Vb')[-1] / equilibrium - 1) < 1e-2


def test_hahnfeldt_regresses_under_continuous_infusion():
    """Test that the inhibitor settles at `dose / dU` and that the vasculature and then the tumour regress."""
    trajectory = _run(HahnfeldtSystem(), until=30.0, dt=0.5)
    assert np.isclose(trajectory.get_values('tumour.U')[-1], 20.0 / 1.3, rtol=1e-4)
    assert np.all(np.diff(trajectory.get_values('tumour.Vb')) < 0)
    assert trajectory.get_values('tumour.A')[-1] < 0.05 * trajectory.get_values('tumour.A')[0]


def test_vascular_core_periphery_inhibitor_restrains_the_tumour():
    """Test that necrosis only accumulates and that the inhibitor keeps the radius, cells, and vessels far lower."""
    untreated = _run(VascularCorePeripherySystem(dose=0.0), until=60.0, dt=1.0)
    treated = _run(VascularCorePeripherySystem(), until=60.0, dt=1.0)
    assert np.all(np.diff(untreated.get_values('tumour.An')) >= 0)
    for column in ('tumour.Ar', 'tumour.Ap', 'tumour.Vbp'):
        assert treated.get_values(column)[-1] < 0.5 * untreated.get_values(column)[-1]
    assert (
        treated.get_values('tumour.Ap')[-1] + treated.get_values('tumour.Ac')[-1]
        < 1e-3 * untreated.get_values('tumour.Ap')[-1]
    )
