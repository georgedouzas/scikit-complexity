"""Test the `_cells` module of `skcomplex.systems.physics.active_matter`."""

import math

import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.systems import create_simulation
from skcomplex.systems.physics.active_matter import szabo_cells


def _order(trajectory, size, start, stop=None):
    """Return the mean normalized velocity of the cells from the displacements between frames, over frames."""
    moves = np.diff(trajectory.values('cells.x'), axis=0)
    moves -= size * np.round(moves / size)
    directions = moves / np.linalg.norm(moves, axis=-1, keepdims=True)
    return float(np.linalg.norm(directions.mean(axis=1), axis=-1)[start:stop].mean())


def test_szabo_cells_order_collectively():
    """Test that the mean normalized velocity grows above 0.8 at a density above the ordering transition."""
    trajectory = create_simulation('szabo_cells').set_params(random_state=1).prepare().run(until=60.0).trajectory_
    size = math.sqrt(200 / 1.0)
    assert _order(trajectory, size, 0, 2) < 0.5
    assert _order(trajectory, size, 80) > 0.8


def test_szabo_cells_stay_disordered_when_dilute():
    """Test that cells far below the critical density reach a smaller order than cells above it."""
    dilute = (
        create_simulation('szabo_cells', density=0.05).set_params(random_state=1).prepare().run(until=60.0).trajectory_
    )
    assert _order(dilute, math.sqrt(200 / 0.05), 80) < 0.8


def test_szabo_cells_repel_and_adhere():
    """Test that a pair closer than the equilibrium distance separates and a pair beyond it approaches."""
    for start, sign in ((0.7, 1), (0.95, -1)):
        system = szabo_cells(2, density=2 / 100.0, eta=0.0, v0=0.0)
        positions = np.array([[5.0, 5.0], [5.0 + start, 5.0]])
        solver = EulerMaruyamaSolver(dt=0.001)
        trajectory = (
            Simulation(system, dt=0.01, solver=solver, initial_state={'cells.x': positions})
            .prepare()
            .run(until=0.01)
            .trajectory_
        )
        gap = trajectory.values('cells.x')[-1]
        assert sign * (gap[1, 0] - gap[0, 0] - start) > 0


def test_szabo_cells_reads_neighbours_within_a_diameter():
    """Test that a pair beyond the diameter of a cell exerts no force."""
    system = szabo_cells(2, density=2 / 100.0, eta=0.0, v0=0.0)
    positions = np.array([[5.0, 5.0], [6.5, 5.0]])
    solver = EulerMaruyamaSolver(dt=0.01)
    final = (
        Simulation(system, dt=0.1, solver=solver, initial_state={'cells.x': positions})
        .prepare()
        .run(until=0.1)
        .trajectory_.values('cells.x')[-1]
    )
    assert np.allclose(final, positions)


def test_szabo_cells_uses_minimum_image_of_the_periodic_box():
    """Test that a pair across the boundary of the box adheres through it."""
    size = 10.0
    system = szabo_cells(2, density=2 / size**2, eta=0.0, v0=0.0)
    positions = np.array([[0.2, 5.0], [9.25, 5.0]])
    solver = EulerMaruyamaSolver(dt=0.001)
    final = (
        Simulation(system, dt=0.01, solver=solver, initial_state={'cells.x': positions})
        .prepare()
        .run(until=0.01)
        .trajectory_.values('cells.x')[-1]
    )
    assert final[0, 0] < 0.2
