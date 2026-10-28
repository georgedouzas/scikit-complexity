"""Test the `_continuous` module of `skcomplex.core.spaces`."""

import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.core.elements import ParticleElement
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver, ScipyODESolver
from skcomplex.errors import ValidationError


@pytest.mark.parametrize(
    ('space', 'message'),
    [
        (ContinuousSpace(extent=(1.0, -1.0)), 'positive side'),
        (ContinuousSpace(extent=(1.0, 1.0), n_dim=3), 'positive side'),
        (ContinuousSpace(), 'number of dimensions'),
        (ContinuousSpace(extent=(1.0,), boundary='sticky'), 'Unknown boundary'),
        (ContinuousSpace(n_dim=2, boundary='periodic'), 'no walls'),
    ],
)
def test_continuous_space_constructor_only_stores_its_arguments(space, message):
    """Test that an invalid space is built without error and rejected by its check."""
    with pytest.raises(ValidationError, match=message):
        space.check()


def test_continuous_space_prepares_its_box():
    """Test that a prepared box has its extent, its dimension, its boundary, and no sites."""
    space = ContinuousSpace(extent=(10.0, 5.0), boundary='periodic').check().prepare()
    assert space.extent_ == (10.0, 5.0)
    assert (space.size_, space.n_dim_, space.shape_, space.periodic_) == (None, 2, (), True)
    assert space.positions_.shape == (0, 2)
    assert space.default_visual_ is None


def test_unbounded_continuous_space_takes_its_dimension():
    """Test that an unbounded space has the given dimension and no extent."""
    space = ContinuousSpace(n_dim=3).check().prepare()
    assert (space.extent_, space.n_dim_, space.periodic_) == ((), 3, False)


def test_continuous_space_has_no_neighbours_of_a_site():
    """Test that a site of a continuous space has no neighbours, since positions are states of the members."""
    assert ContinuousSpace(extent=(1.0, 1.0)).check().prepare()._get_neighbors(0).tolist() == []


def test_continuous_space_clones_from_its_arguments():
    """Test that a clone of a box has the same extent and boundary."""
    space = clone(ContinuousSpace(extent=(2.0, 3.0), boundary='reflecting')).check().prepare()
    assert space.extent_ == (2.0, 3.0)
    assert space.boundary == 'reflecting'
    assert np.array_equal(space.positions_, np.zeros((0, 2)))


def test_reflecting_box_reverses_crossing_particle():
    """Test that a particle crossing a wall comes back into the box with its velocity reversed."""
    system = System(
        elements=[ParticleElement('p', x=[0.9], v=[1.0])],
        space=ContinuousSpace(extent=(1.0,), boundary='reflecting'),
    )
    trajectory = Simulation(system, dt=0.2).check().run(until=0.2).trajectory_
    assert trajectory.values('p.x')[-1, 0] == pytest.approx(0.9)
    assert trajectory.values('p.v')[-1, 0] == pytest.approx(-1.0)


def test_periodic_box_wraps_leaving_particle():
    """Test that a particle leaving through one side enters through the opposite side."""
    system = System(
        elements=[ParticleElement('p', x=[0.9], v=[1.0])],
        space=ContinuousSpace(extent=(1.0,), boundary='periodic'),
    )
    trajectory = Simulation(system, dt=0.2).check().run(until=0.2).trajectory_
    assert trajectory.values('p.x')[-1, 0] == pytest.approx(0.1)
    assert trajectory.values('p.v')[-1, 0] == pytest.approx(1.0)


@pytest.mark.parametrize('solver', [RK4Solver(), ScipyODESolver()], ids=['rk4', 'scipy'])
def test_long_frame_keeps_particle_in_reflecting_box(solver):
    """Test that a frame of duration 5 reflects a particle four times and ends at 0.5 with velocity -1."""
    system = System(
        elements=[ParticleElement('p', x=[0.5], v=[1.0])],
        space=ContinuousSpace(extent=(1.0,), boundary='reflecting'),
    )
    trajectory = Simulation(system, solver=solver, dt=5.0).check().run(until=5).trajectory_
    assert trajectory.values('p.x')[-1, 0] == pytest.approx(0.5, abs=1e-6)
    assert trajectory.values('p.v')[-1, 0] == pytest.approx(-1.0, abs=1e-6)
