"""Test agents of the `_mesa` module of `skcomplex.core.engine.solvers` in continuous spaces and region layers."""

import json
from importlib.resources import files
from typing import Annotated

import numpy as np
import pytest
from shapely import Point, Polygon, box

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import MovementInteraction
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import ContinuousSpace, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import MesaSolver
from skcomplex.core.errors import BackendError, SimulationError, SpaceError
from skcomplex.core.specs import StateSpec
from skcomplex.systems.biology.collective_behaviour import BoidsSystem

RECORDED_BOIDS = 'recorded_boids_two_dimensions.json'


class ProbeElement(Element):
    """Test helper that records what a rule reads of the system and its neighbours."""

    seen: list[object]

    def __init__(self, label, visual=None, *, near: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.near = near

    def _get_laws(self, namespace):
        """Test helper laws."""
        return [RuleLaw(self.read)]

    def read(self, agent, system):
        """Test helper rule that keeps the dimensions and counts the neighbours within radii."""
        ProbeElement.seen.append((agent.id, system.dimensions, len(agent.neighbors(2.0)), len(agent.neighbors(4.0))))


class IdleElement(Element):
    """Test helper agent whose rule does nothing."""

    def _get_laws(self, namespace):
        """Test helper laws."""
        return [RuleLaw(self.idle)]

    def idle(self, agent, system):
        """Test helper rule."""


def _simulate(system, until=0, random_state=0):
    """Return the trajectory of a system on the Mesa solver."""
    return simulate(system, until=until, solver=MesaSolver(), random_state=random_state)


def _concave_polygon():
    """Return an L of three unit squares around a missing fourth, with a hole of area 0.1 in the first square."""
    hole = [(0.2, 0.2), (0.7, 0.2), (0.7, 0.4), (0.2, 0.4)]
    return Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)], holes=[hole]), box(0.2, 0.2, 0.7, 0.4)


def test_recorded_two_dimensional_trajectories_load():
    """Test that the trajectories recorded before three-dimensional spaces load with their frames and shapes."""
    recorded = json.loads((files(__package__) / RECORDED_BOIDS).read_text())
    assert sorted(int(frame) for frame in recorded) == [0, 1, 5, 10, 20]
    assert np.asarray(recorded['20']['locations']).shape == (10, 2)


def test_two_dimensional_space_keeps_recorded_trajectories():
    """Test that a system on a two-dimensional continuous space repeats the trajectories recorded before the change."""
    recorded = json.loads((files(__package__) / RECORDED_BOIDS).read_text())
    for _ in range(2):
        trajectory = simulate(BoidsSystem(10).prepare().system_, until=20, dt=1, random_state=0)
        for frame, values in recorded.items():
            snapshot = trajectory.agents('flock', int(frame))
            assert snapshot.ids.tolist() == values['ids']
            np.testing.assert_allclose(snapshot.locations, values['locations'], rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(snapshot.values['vx'], values['vx'], rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(snapshot.values['vy'], values['vy'], rtol=1e-9, atol=1e-9)


def test_three_dimensional_box_places_agents_inside():
    """Test that the system view reports three dimensions, and agents without a location start inside the box."""
    ProbeElement.seen = []
    group = GroupElement('probes', element=ProbeElement, size=200)
    system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(10.0, 5.0, 2.0)))
    locations = _simulate(system, until=1).agents('probes', 0).locations
    assert ProbeElement.seen[0][1] == (10.0, 5.0, 2.0)
    assert locations.shape == (200, 3)
    assert locations.min() >= 0.0
    assert np.all(locations.max(axis=0) <= (10.0, 5.0, 2.0))
    assert locations[:, 2].max() > 1.0


def test_neighbours_across_the_depth_boundary():
    """Test that two agents across the periodic third side find each other within a radius of 0.5."""

    class NearElement(ProbeElement):
        """Test helper that counts the neighbours within 0.5."""

        def read(self, agent, system):
            """Test helper rule."""
            agent.near = float(len(agent.neighbors(0.5)))

    group = GroupElement('probes', element=NearElement, size=2, location=[(5.0, 5.0, 0.2), (5.0, 5.0, 9.9)])
    system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(10.0, 10.0, 10.0), boundary='periodic'))
    assert _simulate(system, until=1).agents('probes', 1).values['near'].tolist() == [1.0, 1.0]


def test_displacement_takes_the_minimum_image_in_depth():
    """Test that the displacement between agents across the third side wraps every coordinate."""

    class ShiftElement(ProbeElement):
        """Test helper that keeps the depth of the shortest vector to the other agent."""

        def read(self, agent, system):
            """Test helper rule."""
            other = next(item for item in system.agents('probes') if item.id != agent.id)
            agent.near = float(agent.displacement_to(other)[2])

    group = GroupElement('probes', element=ShiftElement, size=2, location=[(5.0, 5.0, 0.2), (5.0, 5.0, 9.9)])
    system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(10.0, 10.0, 10.0), boundary='periodic'))
    near = _simulate(system, until=1).agents('probes', 1).values['near']
    np.testing.assert_allclose(near, [-0.3, 0.3])


def test_depth_counts_in_the_distance():
    """Test that agents that differ by 3 in the third coordinate are neighbours within 4 and not within 2."""
    ProbeElement.seen = []
    group = GroupElement('probes', element=ProbeElement, size=2, location=[(5.0, 5.0, 1.0), (5.0, 5.0, 4.0)])
    _simulate(CompositeSystem(elements=[group], space=ContinuousSpace(extent=(10.0, 10.0, 10.0))), until=1)
    assert [item[2:] for item in ProbeElement.seen] == [(0, 1), (0, 1)]


def test_gaussian_steps_spread_in_depth():
    """Test that the variance of the third coordinate grows under Gaussian steps and positions stay in the box."""
    group = GroupElement('walkers', element=Element, size=100, location=(5.0, 5.0, 5.0))
    walk = MovementInteraction('walk', target='walkers', step='gaussian', scale=0.1)
    system = CompositeSystem(
        elements=[group],
        interactions=[walk],
        space=ContinuousSpace(extent=(10.0, 10.0, 10.0), boundary='periodic'),
    )
    trajectory = _simulate(system, until=100)
    variances = [trajectory.agents('walkers', frame).locations[:, 2].var() for frame in (0, 10, 100)]
    assert variances[0] < variances[1] < variances[2]
    for frame in (1, 50, 100):
        locations = trajectory.agents('walkers', frame).locations
        assert locations.min() >= 0.0
        assert locations.max() <= 10.0


def test_trajectory_reports_three_coordinates_per_agent():
    """Test that a trajectory holds one row of 3 coordinates per agent alive at frame 5, with the agent ids."""
    group = GroupElement('walkers', element=Element, size=7)
    walk = MovementInteraction('walk', target='walkers', step='gaussian', scale=0.1)
    system = CompositeSystem(
        elements=[group],
        interactions=[walk],
        space=ContinuousSpace(extent=(10.0, 10.0, 10.0), boundary='periodic'),
    )
    snapshot = _simulate(system, until=5).agents('walkers', 5)
    assert snapshot.locations.shape == (7, 3)
    assert len(snapshot.ids) == 7


@pytest.mark.parametrize(
    ('space', 'dimension'),
    [
        (ContinuousSpace(extent=(5.0, 5.0, 5.0, 5.0)), '4 dimensions'),
        (ContinuousSpace(n_dim=2), '2 dimensions'),
        (ContinuousSpace(extent=(5.0,)), '1 dimensions'),
    ],
)
def test_unsupported_continuous_space_is_rejected_before_the_first_frame(space, dimension):
    """Test that a space that is no box of two or three sides raises a named error when the solver prepares."""
    group = GroupElement('probes', element=ProbeElement, size=2)
    problem = PassCompiler(solver=MesaSolver()).prepare(CompositeSystem(elements=[group], space=space)).run().problem_
    with pytest.raises(BackendError, match=f"continuous space 'space'.*{dimension}.*two or three dimensions"):
        MesaSolver().prepare(problem)
    with pytest.raises(BackendError, match='two or three dimensions'):
        _simulate(CompositeSystem(elements=[group], space=space), until=1)


@pytest.mark.parametrize(
    'location',
    [(1.0, 2.0), [(1.0, 2.0), (1.0, 2.0), (1.0, 2.0)], box(0.0, 0.0, 1.0, 1.0)],
)
def test_location_of_the_wrong_length_is_reported(location):
    """Test that the compile report holds an initial location that does not fit a three-dimensional space."""
    group = GroupElement('probes', element=ProbeElement, size=3, location=location)
    system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(5.0, 5.0, 5.0)))
    report = PassCompiler().prepare(system).report_
    issue = next(item for item in report.issues if item.code == 'location-dimension')
    assert issue.path == 'probes'
    assert '2 coordinates' in issue.message
    assert '3 dimensions' in issue.message


def test_locations_that_fit_their_space_are_not_reported():
    """Test that fitting positions and polygons in a two-dimensional space leave the compile report clean."""
    for location in ((1.0, 2.0, 3.0), [(1.0, 2.0, 3.0)] * 2):
        group = GroupElement('probes', element=ProbeElement, size=2, location=location)
        system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(5.0, 5.0, 5.0)))
        assert PassCompiler().prepare(system).report_.ok
    group = GroupElement('probes', element=ProbeElement, size=2, location=[box(0.0, 0.0, 1.0, 1.0)] * 2)
    system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(5.0, 5.0)))
    assert PassCompiler().prepare(system).report_.ok


def test_move_to_of_the_wrong_length_raises_space_error():
    """Test that a move to a location that does not fit the space raises a named error during the run."""
    group = GroupElement('walkers', element=Element, size=1, location=(1.0, 1.0, 1.0))
    walk = MovementInteraction('walk', target='walkers', step=lambda agent, system: (2.0, 2.0))
    system = CompositeSystem(elements=[group], interactions=[walk], space=ContinuousSpace(extent=(5.0, 5.0, 5.0)))
    with pytest.raises(SpaceError, match='has 2 coordinates, and the space has 3 dimensions'):
        _simulate(system, until=1)


def test_agents_per_census_region():
    """Test that members start inside the polygons given as their locations."""
    regions = [box(0.0, 0.0, 4.0, 4.0), box(5.0, 5.0, 9.0, 9.0), box(0.0, 5.0, 4.0, 9.0)]
    group = GroupElement('people', element=IdleElement, size=150, location=[regions[0]] * 100 + [regions[1]] * 50)
    system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(10.0, 10.0), regions={'census': regions}))
    snapshot = _simulate(system).agents('people', 0)
    assert all(regions[0].contains(Point(point)) for point in snapshot.locations[:100])
    assert all(regions[1].contains(Point(point)) for point in snapshot.locations[100:])


def test_one_polygon_is_shared_by_every_member():
    """Test that one polygon as the location of a group places every member at its own point inside it."""
    region = box(1.0, 1.0, 2.0, 2.0)
    group = GroupElement('people', element=IdleElement, size=5, location=region)
    system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(5.0, 5.0)))
    locations = _simulate(system).agents('people', 0).locations
    assert all(region.contains(Point(point)) for point in locations)
    assert len({tuple(point) for point in locations}) == 5


def test_region_placement_is_reproducible():
    """Test that two runs with one random state start the agents at the same points, and another seed differs."""

    def start(random_state):
        group = GroupElement('people', element=IdleElement, size=20, location=box(0.0, 0.0, 4.0, 4.0))
        system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(5.0, 5.0)))
        return _simulate(system, random_state=random_state).agents('people', 0).locations

    np.testing.assert_array_equal(start(0), start(0))
    assert not np.array_equal(start(0), start(1))


def test_placement_is_uniform_in_a_concave_polygon_with_a_hole():
    """Test that 20000 agents lie in an L with a hole, with the share of each unit square equal to its area share."""
    polygon, hole = _concave_polygon()
    group = GroupElement('people', element=IdleElement, size=20000, location=polygon)
    system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(2.0, 2.0)))
    locations = _simulate(system).agents('people', 0).locations
    assert all(polygon.contains(Point(point)) for point in locations[:2000])
    x, y = locations[:, 0], locations[:, 1]
    assert not np.any((x > 0.2) & (x < 0.7) & (y > 0.2) & (y < 0.4))
    assert not np.any((x > 1.0) & (y > 1.0))
    shares = [np.mean((x < 1) & (y < 1)), np.mean((x >= 1) & (y < 1)), np.mean((x < 1) & (y >= 1))]
    np.testing.assert_allclose(shares, np.array([0.9, 1.0, 1.0]) / polygon.area, atol=0.02)
    assert hole.area == pytest.approx(0.1)


def test_move_to_a_favourite_place():
    """Test that a step returning the polygon of a region moves the agent into it, and the agent finds it there."""
    places = [box(0.0, 0.0, 2.0, 2.0), box(3.0, 3.0, 5.0, 5.0), box(4.0, 4.0, 6.0, 6.0)]
    found = []

    def to_favourite(agent, system):
        found.append(system.regions('places'))
        return system.region('places', 1)

    def read(agent, system):
        found.append(agent.regions_here('places'))

    class ReaderElement(Element):
        """Test helper that reads the regions of its agent."""

        def _get_laws(self, namespace):
            """Test helper laws."""
            return [RuleLaw(read)]

    group = GroupElement('people', element=ReaderElement, size=30, location=places[0])
    move = MovementInteraction('move', target='people', step=to_favourite, schedule='member')
    system = CompositeSystem(
        elements=[group],
        interactions=[move],
        space=ContinuousSpace(extent=(10.0, 10.0), regions={'places': places}),
    )
    locations = _simulate(system, until=2).agents('people', 2).locations
    assert all(places[1].contains(Point(point)) for point in locations)
    assert 3 in found
    assert [0] in found
    assert any(1 in item for item in found if isinstance(item, list))


def test_created_agent_is_placed_in_a_polygon():
    """Test that a created agent starts inside the polygon of its location."""
    region = box(1.0, 1.0, 2.0, 2.0)

    class SpawnerElement(Element):
        """Test helper that creates one agent in a region."""

        def _get_laws(self, namespace):
            """Test helper laws."""
            return [RuleLaw(self.spawn)]

        def spawn(self, agent, system):
            """Test helper rule."""
            if system.count('people') == 1:
                system.create('people', location=region)

    group = GroupElement('people', element=SpawnerElement, size=1, location=(4.0, 4.0), variable=True)
    system = CompositeSystem(elements=[group], space=ContinuousSpace(extent=(5.0, 5.0)))
    locations = _simulate(system, until=1).agents('people', 1).locations
    assert locations.shape == (2, 2)
    assert region.contains(Point(locations[1]))


@pytest.mark.parametrize(
    ('step', 'error', 'message'),
    [
        (lambda agent, system: agent.regions_here('absent'), SimulationError, 'no region layer'),
        (lambda agent, system: system.region('places', 5), SimulationError, 'has 1 regions'),
        (lambda agent, system: Polygon([(0, 0), (1, 1), (1, 0), (0, 1)]), SpaceError, 'invalid polygon'),
    ],
)
def test_region_queries_and_invalid_polygons_raise_named_errors(step, error, message):
    """Test that an unknown layer, an unknown region, and an invalid polygon raise errors that say what is wrong."""
    group = GroupElement('people', element=IdleElement, size=1, location=(1.0, 1.0))
    move = MovementInteraction('move', target='people', step=step)
    space = ContinuousSpace(extent=(5.0, 5.0), regions={'places': [box(0.0, 0.0, 1.0, 1.0)]})
    with pytest.raises(error, match=message):
        _simulate(CompositeSystem(elements=[group], interactions=[move], space=space), until=1)


def test_polygon_is_no_location_of_a_lattice():
    """Test that a polygon as the location of an agent on a lattice raises a named error."""
    group = GroupElement('people', element=IdleElement, size=1, location=box(0.0, 0.0, 1.0, 1.0))
    with pytest.raises(SpaceError, match='location of a continuous space'):
        _simulate(CompositeSystem(elements=[group], space=LatticeSpace((5, 5))))
