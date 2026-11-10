"""Test the `_cell_kinetics` module of `skcomplex.systems.biology.oncology`."""

import numpy as np

from skcomplex.systems.biology.oncology import CellCycleChemotherapySystem, ErythropoiesisMercaptopurineSystem

from .conftest import _run


def _growth_rate(trajectory, names):
    """Return the late exponential growth rate of the total cells."""
    total = np.log(sum(trajectory.get_values(f'tumour.{name}') for name in names))
    late = trajectory.times >= 300.0
    return np.polyfit(trajectory.times[late], total[late], 1)[0]


def test_cell_cycle_chemotherapy_kills_above_half():
    """Test that the tumour grows at the dominant eigenvalue of its cycle, which changes sign at a kill of one half."""
    l1, l2, l3 = 0.197, 0.395, 0.107
    for u1, u2 in ((0.4, 0.5), (0.6, 0.0)):
        matrix = np.array(
            [[-l1, 0.0, 2 * (1 - u1) * l3], [l1, -(1 - u2) * l2, 0.0], [0.0, (1 - u2) * l2, -l3]],
        )
        trajectory = _run(CellCycleChemotherapySystem(u1=u1, u2=u2), until=400.0, dt=2.0)
        rate = _growth_rate(trajectory, ('A1', 'A2', 'A3'))
        assert abs(rate - np.linalg.eigvals(matrix).real.max()) < 1e-6
        assert (rate > 0) == (u1 < 0.5)
    dormant = _run(CellCycleChemotherapySystem(dormant=True, u1=0.55, v=1.0), until=400.0, dt=2.0)
    assert _growth_rate(dormant, ('A0', 'A1', 'A2')) < 0


def test_erythropoiesis_mercaptopurine_steady_volume():
    """Test that the mean volume settles at `MCV0 + g1 U` and the mature cells at `r1 Q / d1`."""
    trajectory = _run(ErythropoiesisMercaptopurineSystem(U=500.0), until=6000.0, dt=20.0)
    assert abs(trajectory.get_values('patient.MCV')[-1] / (85.0 + 0.0317 * 500.0) - 1) < 1e-3
    assert (
        abs(trajectory.get_values('patient.L')[-1] / (0.00139 * trajectory.get_values('patient.Q')[-1] / 0.0071) - 1)
        < 1e-3
    )
    assert trajectory.get_values('patient.L')[-1] < 0.9 * trajectory.get_values('patient.L')[0]
