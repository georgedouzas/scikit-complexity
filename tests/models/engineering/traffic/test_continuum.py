"""Test the `_continuum` module of `skcomplex.models.engineering.traffic`."""

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.formalisms.fields import SSPRK
from skcomplex.models.engineering.traffic import (
    kerner_konhauser,
    lighthill_whitham_richards,
    payne,
)


def test_lighthill_whitham_richards_shock_speed():
    """Test that the shock between free and queued traffic moves at the Rankine-Hugoniot speed."""
    trajectory = Simulation(lighthill_whitham_richards(), dt=0.05, solver=SSPRK()).run(until=1.0).trajectory_
    density = trajectory['road.rho'][-1]
    centers = (np.arange(200) + 0.5) / 200
    assert centers[np.argmax(np.diff(density))] == pytest.approx(0.5 + (1 - 1.1), abs=0.01)


def test_payne_conserves_cars_and_keeps_equilibrium():
    """Test that the Payne model conserves the cars and keeps uniform flow at the equilibrium speed."""
    trajectory = Simulation(payne(50), dt=0.1, solver=SSPRK()).run(until=0.5).trajectory_
    assert trajectory['road.rho'].sum(axis=1) == pytest.approx(np.full(6, trajectory['road.rho'][0].sum()))
    uniform = Simulation(payne(50, bump=0.0), dt=0.1, solver=SSPRK()).run(until=0.5).trajectory_
    assert np.ptp(uniform['road.q'][-1]) < 1e-12


def test_kerner_konhauser_forms_a_cluster():
    """Test that a small bump of density grows into a dense cluster while the cars are conserved."""
    trajectory = Simulation(kerner_konhauser(), dt=10.0, solver=SSPRK(cfl=0.15)).run(until=40).trajectory_
    density = trajectory['road.rho']
    assert density[-1].max() > 0.35
    assert density.sum(axis=1)[-1] == pytest.approx(density.sum(axis=1)[0])
