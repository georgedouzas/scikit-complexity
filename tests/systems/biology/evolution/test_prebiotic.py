"""Test the `_prebiotic` module of `skcomplex.systems.biology.evolution`."""

import numpy as np
import pytest

from skcomplex.systems.biology.evolution import (
    eigen_quasispecies,
    hypercycles,
)

from ..conftest import _run_until
from .conftest import PRECISE


@pytest.mark.parametrize(('u', 'master'), [(0.3, 0.4), (0.6, 0.0)])
def test_quasispecies_error_threshold(u, master):
    """Test that the master sequence keeps the share `1 - u / sigma` below the error threshold and is lost above."""
    trajectory = _run_until(eigen_quasispecies(u=u, sigma=0.5), 200, dt=1)
    assert trajectory.values('master.x')[-1] == pytest.approx(master, abs=1e-3)
    total = sum(trajectory.values(f'{label}.x')[-1] for label in ['master', *(f'k{k}' for k in range(1, 13))])
    assert total == pytest.approx(1)


def test_hypercycle_keeps_all_and_network_selects():
    """Test that a hypercycle of 4 molecules keeps the simplex and every molecule, and a complete network selects."""
    system = hypercycles(4)
    assert system.elements[0].size == 4
    cycle = _run_until(system, 1500, dt=10.0, solver=PRECISE)
    concentrations = cycle.values('soup.x')
    assert np.abs(concentrations.sum(axis=1) - 1).max() < 1e-10
    assert concentrations[-1].min() > 0.01
    assert np.abs(concentrations[-1] - concentrations[-2]).max() < 1e-4
    complete = _run_until(hypercycles(20, alpha=1.0, omega=50.0, topology='complete'), 50, dt=1.0)
    assert abs(np.sum(complete.values('soup.x')[-1] > 1e-3) - 10) <= 1


def test_hypercycle_of_five_loses_members():
    """Test that a hypercycle of five molecules loses some members and settles at a fixed point."""
    concentrations = _run_until(hypercycles(5), 400, dt=0.5).values('soup.x')
    assert concentrations[-1].min() < 1e-3
    assert np.abs(concentrations[-1] - concentrations[-2]).max() < 1e-4
