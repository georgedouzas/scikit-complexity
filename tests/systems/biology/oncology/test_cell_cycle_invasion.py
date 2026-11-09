"""Test the `_cell_cycle_invasion` module of `skcomplex.systems.biology.oncology`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.experiments import calibrate
from skcomplex.systems.biology.oncology import CellCycleInvasionSystem

PHASES = ('g1', 'es', 'sg2m')


def _cells(trajectory):
    """Return the occupation of every site at every frame."""
    return sum(np.asarray(trajectory.get_values(f'cells.{phase}')) for phase in PHASES)


def test_cells_invade_the_scratch():
    """Test that the number of cells inside the scratch grows over 48 hours."""
    system = CellCycleInvasionSystem(24, 12)
    trajectory = simulate(system, until=48, random_state=0)
    columns = np.arange(24 * 12) % 24
    inside = (columns >= 9) & (columns < 15)
    occupied = _cells(trajectory)
    assert occupied[0][inside].sum() == 0
    assert occupied[-1][inside].sum() > 5
    assert np.all((occupied == 0) | (occupied == 1))


def test_full_lattice_keeps_its_cells_in_g1():
    """Test that a full lattice without moves keeps its cells and no cell leaves G1."""
    system = CellCycleInvasionSystem(10, 6, Mr=0.0, My=0.0, Mg=0.0, scratch=None)
    trajectory = simulate(system, until=48, random_state=0)
    occupied = _cells(trajectory)
    assert np.all(occupied.sum(axis=1) == 60)
    g1 = trajectory.get_values('g1_cells')
    assert np.all(g1 == g1[0])
    assert trajectory.get_values('es_cells')[-1] < trajectory.get_values('es_cells')[0]


def test_phases_are_observables():
    """Test that the cells per phase are observables that add up to the occupied sites."""
    trajectory = simulate(CellCycleInvasionSystem(12, 6), until=10, random_state=1)
    total = sum(trajectory.get_values(f'{phase}_cells') for phase in PHASES)
    np.testing.assert_array_equal(total, _cells(trajectory).sum(axis=1))


@pytest.mark.slow
def test_abc_smc_compares_the_three_phase_counts():
    """Test that ABC-SMC with the final counts of the three phases as summaries compares three statistics."""
    system = CellCycleInvasionSystem(12, 6, Rr=0.1)
    trajectory = simulate(system, until=24, random_state=0)
    times = np.arange(0.0, 25.0, 6.0)
    data = {
        f'{phase}_cells': (times, np.interp(times, trajectory.times, trajectory.get_values(f'{phase}_cells')))
        for phase in PHASES
    }

    def finals(series):
        return [series[f'{phase}_cells'][-1] for phase in PHASES]

    result = calibrate(
        CellCycleInvasionSystem(12, 6),
        data,
        {'Rr': (0.01, 0.5)},
        method='abc_smc',
        summary=finals,
        n_samples=40,
        budget=200,
        random_state=0,
    )
    assert result.diagnostics['summaries'] == 3
    assert 0.01 <= result.estimates['Rr'] <= 0.5
