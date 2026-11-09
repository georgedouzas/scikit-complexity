"""Test the `_continuous` module of `skcomplex.core.components.spaces`."""

from typing import Annotated

import numpy as np
import numpy.typing as npt
import pytest
from shapely import Point, Polygon, box
from sklearn.base import clone

from skcomplex.core.components.elements import Element
from skcomplex.core.components.spaces import BaseSpace, ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver, ScipyODESolver
from skcomplex.core.errors import SpaceError, ValidationError
from skcomplex.core.specs import StateSpec, VisualSpec


class BallElement(Element):
    """Point of a continuous space, with a vector state `x` of order 2."""

    def __init__(
        self,
        label: str,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, order=2, role='position')] = 0.0,
        x_d1: npt.ArrayLike = 0.0,
        visual: VisualSpec | None = None,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Set `default_visual_`, points at the values of `x`."""
        super()._prepare(context)
        self.default_visual_ = VisualSpec('points', {'position': 'x'})


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
        space.prepare()


def test_continuous_space_prepares_its_box():
    """Test that a prepared box has its extent, its dimension, its boundary, and no sites."""
    space = ContinuousSpace(extent=(10.0, 5.0), boundary='periodic').prepare()
    assert space.extent_ == (10.0, 5.0)
    assert (space.size_, space.n_dim_, space.shape_, space.periodic_) == (None, 2, (), True)
    assert space.positions_.shape == (0, 2)
    assert space.default_visual_ is None


def test_unbounded_continuous_space_takes_its_dimension():
    """Test that an unbounded space has the given dimension and no extent."""
    space = ContinuousSpace(n_dim=3).prepare()
    assert (space.extent_, space.n_dim_, space.periodic_) == ((), 3, False)


def test_continuous_space_has_no_neighbours_of_a_site():
    """Test that a site of a continuous space has no neighbours, since positions are states of the members."""
    assert ContinuousSpace(extent=(1.0, 1.0)).prepare()._get_neighbors(0).tolist() == []


def test_continuous_space_clones_from_its_arguments():
    """Test that a clone of a box has the same extent and boundary."""
    space = clone(ContinuousSpace(extent=(2.0, 3.0), boundary='reflecting')).prepare()
    assert space.extent_ == (2.0, 3.0)
    assert space.boundary == 'reflecting'
    assert np.array_equal(space.positions_, np.zeros((0, 2)))


def test_continuous_space_indexes_its_regions():
    """Test that the derived index of a layer of three regions answers which of them contains a point."""
    regions = [box(0.0, 0.0, 1.0, 1.0), box(2.0, 0.0, 3.0, 1.0), box(0.0, 2.0, 1.0, 3.0)]
    space = ContinuousSpace(extent=(4.0, 4.0), regions={'census': regions}).prepare()
    assert space.regions['census'] is regions
    assert space.regions_['census'].query(Point(2.5, 0.5), predicate='intersects').tolist() == [1]
    assert space.regions_['census'].query(Point(3.5, 3.5), predicate='intersects').tolist() == []
    assert ContinuousSpace(extent=(4.0, 4.0)).prepare().regions_ == {}


def test_continuous_space_clones_with_its_regions():
    """Test that a clone of a space keeps its layers and indexes them."""
    space = clone(ContinuousSpace(extent=(4.0, 4.0), regions={'parks': [box(0.0, 0.0, 1.0, 1.0)]}))
    assert list(space.prepare().regions_) == ['parks']


@pytest.mark.parametrize(
    ('region', 'message'),
    [
        (Polygon([(0, 0), (1, 1), (1, 0), (0, 1)]), "region 1 of the layer 'places' is not a valid polygon"),
        (Polygon(), "region 1 of the layer 'places' is not a valid polygon"),
        (Point(0.0, 0.0), "region 1 of the layer 'places' is <POINT"),
        (Polygon([(0, 0, 1), (1, 0, 1), (1, 1, 1)]), "region 1 of the layer 'places' is <POLYGON Z"),
    ],
)
def test_continuous_space_rejects_invalid_regions(region, message):
    """Test that preparing a space with an invalid region raises an error that names the layer and the region."""
    space = ContinuousSpace(extent=(4.0, 4.0), regions={'places': [box(0.0, 0.0, 1.0, 1.0), region]})
    with pytest.raises(SpaceError, match=message):
        space.prepare()


def test_continuous_space_keeps_regions_in_two_dimensions():
    """Test that a space with regions and three sides raises an error that names the layer and the dimension."""
    space = ContinuousSpace(extent=(4.0, 4.0, 4.0), regions={'places': [box(0.0, 0.0, 1.0, 1.0)]})
    with pytest.raises(SpaceError, match=r'places.*3 dimensions'):
        space.prepare()


def test_reflecting_box_reverses_crossing_particle():
    """Test that a particle crossing a wall comes back into the box with its velocity reversed."""
    system = CompositeSystem(
        elements=[BallElement('p', x=[0.9], x_d1=[1.0])],
        space=ContinuousSpace(extent=(1.0,), boundary='reflecting'),
    )
    trajectory = simulate(system, until=0.2, dt=0.2)
    assert trajectory.values('p.x')[-1, 0] == pytest.approx(0.9)
    assert trajectory.values('p.x_d1')[-1, 0] == pytest.approx(-1.0)


def test_periodic_box_wraps_leaving_particle():
    """Test that a particle leaving through one side enters through the opposite side."""
    system = CompositeSystem(
        elements=[BallElement('p', x=[0.9], x_d1=[1.0])],
        space=ContinuousSpace(extent=(1.0,), boundary='periodic'),
    )
    trajectory = simulate(system, until=0.2, dt=0.2)
    assert trajectory.values('p.x')[-1, 0] == pytest.approx(0.1)
    assert trajectory.values('p.x_d1')[-1, 0] == pytest.approx(1.0)


@pytest.mark.parametrize('solver', [RK4Solver(), ScipyODESolver()], ids=['rk4', 'scipy'])
def test_long_frame_keeps_particle_in_reflecting_box(solver):
    """Test that a frame of duration 5 reflects a particle four times and ends at 0.5 with velocity -1."""
    system = CompositeSystem(
        elements=[BallElement('p', x=[0.5], x_d1=[1.0])],
        space=ContinuousSpace(extent=(1.0,), boundary='reflecting'),
    )
    trajectory = simulate(system, until=5, solver=solver, dt=5.0)
    assert trajectory.values('p.x')[-1, 0] == pytest.approx(0.5, abs=1e-6)
    assert trajectory.values('p.x_d1')[-1, 0] == pytest.approx(-1.0, abs=1e-6)
