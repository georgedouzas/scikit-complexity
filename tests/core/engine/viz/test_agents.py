"""Test the `_agents` module of `skcomplex.core.engine.viz`."""

import numpy as np
import pytest
import rerun as rr
from shapely import Polygon

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.viz import AgentsVisual
from skcomplex.core.specs import AgentSnapshotSpec
from skcomplex.systems.social.organizations import ObesitySystem

from .conftest import Logs, _frames, _radius


class RestingElement(Element):
    """Test helper agent whose rule does nothing."""

    def _get_laws(self, namespace):
        """Test helper laws."""
        return [RuleLaw(self.rest)]

    def rest(self, agent, system):
        """Test helper rule."""


def test_agents_visual_radius_scales_with_continuous_space(log_agents):
    """Test that agents in a continuous space take a share of its side, and agents on a grid 0.4 cells."""
    problem, frames = _frames(ObesitySystem(n_people=10, n_places=4, random_state=0).prepare().system_, steps=1, dt=1.0)
    space = problem.agent_space
    assert isinstance(space, ContinuousSpace)
    logs = Logs()
    snapshot = frames[0].agents['people']
    log_agents(AgentsVisual(), logs, 'people', [snapshot], space)
    assert _radius(logs.of('people')[0]) == pytest.approx(0.02 * 280.0)
    log_agents(AgentsVisual(), logs, 'grid', [snapshot])
    assert _radius(logs.of('grid')[0]) == pytest.approx(0.4)


def test_agents_visual_widens_color_range_of_equal_start(log_agents):
    """Test that agents that all start with one value get different colors once their values spread."""
    visual, logs = AgentsVisual(color_by='rank'), Logs()
    space = ContinuousSpace(extent=(10.0, 10.0)).prepare()
    first = AgentSnapshotSpec(np.arange(3), {'rank': np.zeros(3)}, np.zeros((3, 2)))
    later = AgentSnapshotSpec(np.arange(3), {'rank': np.array([0.0, 1.0, 2.0])}, np.zeros((3, 2)))
    log_agents(visual, logs, 'animals', [first, later], space)
    colors = logs.of('animals')[-1].colors.as_arrow_array().to_pylist()
    assert len(set(colors)) == 3
    assert visual.color_range is None


def test_agents_visual_draws_continuous_spaces_with_vertical_axis_up(log_agents):
    """Test that agents in a continuous space are logged with y negated and the view covers the whole space."""
    space = ContinuousSpace(extent=(10.0, 20.0)).prepare()
    snapshot = AgentSnapshotSpec(np.arange(1), {'rank': np.zeros(1)}, np.array([[3.0, 4.0]]))
    logs = Logs()
    log_agents(AgentsVisual(), logs, 'birds', [snapshot], space)
    assert logs.of('birds')[0].positions.as_arrow_array().to_pylist() == [[3.0, -4.0]]
    problem, _ = _frames(ObesitySystem(n_people=10, n_places=4, random_state=0).prepare().system_, steps=1, dt=1.0)
    view = AgentsVisual(path=problem.agents[0].path).prepare(problem).views_[0]
    bounds = view.properties['VisualBounds2D']
    assert bounds is not None


def test_agents_visual_draws_boids_in_three_dimensions(log_agents):
    """Test that agents in a three-dimensional space are logged as one `Points3D` batch per frame at their positions."""
    space = ContinuousSpace(extent=(10.0, 20.0, 30.0)).prepare()
    locations = [np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]), np.array([[3.5, 4.5, 5.5], [6.5, 7.5, 8.5]])]
    snapshots = [AgentSnapshotSpec(np.arange(2), {'rank': np.zeros(2)}, points) for points in locations]
    logs = Logs()
    log_agents(AgentsVisual(), logs, 'birds', snapshots, space)
    batches = logs.of('birds')
    assert [type(batch).__name__ for batch in batches] == ['Points3D', 'Points3D']
    assert batches[1].positions.as_arrow_array().to_pylist() == locations[1].tolist()
    recording = rr.RecordingStream('three_dimensions')
    stream = recording.memory_recording()
    log_agents(AgentsVisual(), recording, 'birds', snapshots, space)
    assert len(stream.drain_as_bytes()) > 0


def test_agents_visual_has_a_three_dimensional_view():
    """Test that the view of agents in a three-dimensional space is a 3D spatial view with their path as origin."""
    flock = GroupElement('flock', element=RestingElement, size=2)
    system = CompositeSystem(elements=[flock], space=ContinuousSpace(extent=(5.0, 5.0, 5.0)))
    problem = PassCompiler().prepare(system).run().problem_
    view = AgentsVisual(path='flock').prepare(problem).views_[0]
    assert type(view).__name__ == 'Spatial3DView'


def test_agents_visual_draws_regions_once_under_the_agents(log_agents):
    """Test that a boundary layer is logged once as static strips, and the agents per frame as `Points2D`."""
    hole = [(2.5, 0.5), (3.5, 0.5), (3.5, 1.5)]
    island = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0)], holes=[hole])
    space = ContinuousSpace(extent=(5.0, 5.0), regions={'boundary': [island]}).prepare()
    snapshots = [AgentSnapshotSpec(np.arange(1), {'rank': np.zeros(1)}, np.array([[3.0, 1.0]]))] * 3
    logs = Logs()
    log_agents(AgentsVisual(), logs, 'people', snapshots, space)
    assert logs.static == ['people/regions/boundary']
    strips = logs.of('people/regions/boundary')
    assert [type(item).__name__ for item in strips] == ['LineStrips2D']
    assert strips[0].strips.as_arrow_array().to_pylist()[0][1] == [4.0, 0.0]
    assert strips[0].strips.as_arrow_array().to_pylist()[0][2] == [4.0, -4.0]
    assert len(strips[0].strips.as_arrow_array().to_pylist()) == 2
    assert [type(item).__name__ for item in logs.of('people')] == ['Points2D'] * 3
    recording = rr.RecordingStream('island')
    stream = recording.memory_recording()
    log_agents(AgentsVisual(), recording, 'people', snapshots, space)
    assert len(stream.drain_as_bytes()) > 0
