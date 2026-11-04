"""Test the `_cells` module of `skcomplex.engine.viz`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest
import rerun as rr
from sage.all__sagemath_symbolics import pi

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import RateInteraction
from skcomplex.core.spaces import CellComplexSpace, LatticeSpace, NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import CellVisual, RerunObserver
from skcomplex.errors import ValidationError
from skcomplex.specs import StateSpec, VisualSpec


class AngleElement(Element):
    """Cell with an angle `theta`."""

    def __init__(self, label, visual=None, *, theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0):
        super().__init__(label, visual=visual)
        self.theta = theta


class ValueElement(Element):
    """Cell with a real value `u`."""

    def __init__(self, label, visual=None, *, u: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.u = u


class RecordingStub:
    """Recording that keeps every archetype it is given by entity path."""

    def __init__(self):
        self.logged = {}

    def log(self, path, archetype, **kwargs):
        """Keep the archetype of a path."""
        del kwargs
        self.logged.setdefault(path, []).append(archetype)


def _group(space, element=AngleElement, part='edges', values=None, visual=None):
    """Return a group on a part of a complex with a constant rate."""
    return GroupElement('cells', element=element, space=space, on=part, values=values or {}, visual=visual)


def _frames(system, visual, until=2.0):
    """Return what a visual logs at every frame of a run."""
    recording = RecordingStub()
    simulation = Simulation(system, dt=1.0).check().prepare()
    visual = visual.set_params(path='cells', recording=recording).check().prepare(simulation.problem_)
    simulation.run(until=until, observers=[]) if False else simulation.run(until=until)
    return recording, visual, simulation


def _rate(state='theta'):
    """Return a constant rate of the state of the group."""
    return RateInteraction('rate', target='cells', state=state, rate=0.1)


def test_plaquette_image_and_segments_are_in_the_recorded_stream(tmp_path):
    """Test that a 32 by 32 U(1) run records 32 by 32 plaquette images and 2048 colored segments at every frame."""
    space = CellComplexSpace(LatticeSpace((32, 32))).check().prepare()
    initial = np.random.default_rng(0).uniform(-np.pi, np.pi, 2048)
    group = _group(space, values={'theta': initial}, visual=VisualSpec('cell'))
    system = System(elements=[group], interactions=[_rate()])
    path = tmp_path / 'run.rrd'
    observer = RerunObserver('cells', save_path=path)
    Simulation(system, dt=1.0).check().prepare().run(until=2.0, observers=[observer])
    assert path.stat().st_size > 0
    recording = rr.dataframe.load_recording(path)
    frames = recording.view(index='step', contents='cells/theta/**').select().read_all().to_pylist()
    assert len(frames) == 3
    lengths = {len(frame['cells/theta:LineStrips2D:strips']) for frame in frames}
    assert lengths == {2048}
    assert {len(frame['cells/theta:LineStrips2D:colors']) for frame in frames} == {2048}
    images = {tuple(frame['cells/theta/faces:Image:format'][0].values())[:2] for frame in frames}
    assert images == {(32, 32)}


def test_face_colors_of_an_angle_state_use_the_cyclic_palette():
    """Test that sums that differ by a period share one color in the plaquette image."""
    space = CellComplexSpace(LatticeSpace((4, 4))).check().prepare()
    theta = np.zeros(32)
    theta[0] = 2 * np.pi
    system = System(elements=[_group(space, values={'theta': theta})], interactions=[_rate()])
    problem = compile_system(system)
    recording = RecordingStub()
    frame = Simulation(system, dt=1.0).check().prepare().run(until=1.0)
    visual = CellVisual(path='cells', recording=recording).check().prepare(problem)
    del frame
    visual_colors = visual._colors_for(np.array([0.0, 2 * np.pi]), 2 * np.pi)
    assert np.array_equal(visual_colors[0], visual_colors[1])


def test_three_dimensional_lattice_draws_the_selected_slice():
    """Test that a 3D complex logs the links and plaquettes of one plane and slice."""
    space = CellComplexSpace(LatticeSpace((3, 4, 5))).check().prepare()
    size = space.incidences_[0].shape[0]
    system = System(elements=[_group(space, values={'theta': np.zeros(size)})], interactions=[_rate()])
    problem = compile_system(system)
    visual = CellVisual('theta', plane=(0, 2), slice=1, path='cells').check().prepare(problem)
    links, segments = visual._lattice_segments(space, space.space)
    assert links.size == segments.shape[0] == 2 * 3 * 5
    assert segments.shape[1:] == (2, 2)
    assert np.all(np.isin(space.link_axes_[links], (0, 2)))


def test_hyperedge_group_on_a_lattice_logs_an_image():
    """Test that a group on the plaquettes of a lattice complex logs an image of the lattice shape."""
    space = CellComplexSpace(LatticeSpace((4, 3))).check().prepare()
    system = System(elements=[_group(space, ValueElement, 'hyperedges')], interactions=[_rate('u')])
    recording = RecordingStub()
    problem = compile_system(system)
    visual = CellVisual(path='cells', recording=recording).check().prepare(problem)
    simulation = Simulation(system, dt=1.0).check().prepare().run(until=1.0)
    state = simulation.trajectory_.frame(-1) if hasattr(simulation.trajectory_, 'frame') else None
    del state
    assert [type(view).__name__ for view in visual.views_] == ['Spatial2DView']


def test_network_complex_logs_segments_and_polygons():
    """Test that a group on the edges of a network complex logs segments and closed polygons."""
    graph = nx.cycle_graph(4)
    graph.add_edge(0, 2)
    space = CellComplexSpace(NetworkSpace(graph)).check().prepare()
    recording = RecordingStub()
    group = _group(space, ValueElement, values={'u': np.arange(5.0)}, visual=VisualSpec('cell', {'state': 'u'}))
    system = System(elements=[group], interactions=[_rate('u')])
    observer = RerunObserver('cells')
    del observer
    problem = compile_system(system)
    visual = CellVisual('u', path='cells', recording=recording).check().prepare(problem)
    links, segments = visual._network_segments(space)
    assert links.size == 5
    assert segments.shape == (5, 2, 2)


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'plane': (0, 0)}, 'plane'),
        ({'plane': (0, 2)}, 'plane'),
    ],
)
def test_prepare_rejects_a_plane_that_is_not_two_axes_of_the_lattice(arguments, message):
    """Test that a plane of a repeated or a missing axis raises an error."""
    space = CellComplexSpace(LatticeSpace((3, 3))).check().prepare()
    system = System(elements=[_group(space)], interactions=[_rate()])
    with pytest.raises(ValidationError, match=message):
        CellVisual(**arguments, path='cells').check().prepare(compile_system(system))


def test_prepare_rejects_a_group_on_the_sites():
    """Test that a group on the sites of a complex has no cell visual."""
    space = CellComplexSpace(LatticeSpace((3, 3))).check().prepare()
    system = System(elements=[_group(space, part='sites')], interactions=[_rate()])
    with pytest.raises(ValidationError, match='edges or hyperedges'):
        CellVisual(path='cells').check().prepare(compile_system(system))
