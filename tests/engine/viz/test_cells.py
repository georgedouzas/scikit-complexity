"""Test the `_cells` module of `skcomplex.engine.viz`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import pi

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import RateInteraction
from skcomplex.core.spaces import CellComplexSpace, LatticeSpace, NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.problems import FrameSpec
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import CellVisual, RerunObserver
from skcomplex.errors import ValidationError
from skcomplex.specs import StateSpec, VisualSpec

SIDE = 32
SEGMENTS = 2 * SIDE * SIDE
COLOR_CHANNELS = 3


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
    """Recording that keeps every archetype it receives by entity path."""

    def __init__(self):
        self.logged = {}

    def log(self, path, archetype, static=False):
        """Keep the archetype of a path."""
        del static
        self.logged.setdefault(path, []).append(archetype)


def _system(space, element=AngleElement, part='edges', values=None, visual=None):
    """Return a system of a group on a part of a complex, one of whose states grows at a constant rate."""
    group = GroupElement('cells', element=element, space=space, on=part, values=values or {}, visual=visual)
    state = 'theta' if element is AngleElement else 'u'
    return System(elements=[group], interactions=[RateInteraction('rate', target='cells', state=state, rate=0.1)])


def _logged(system, visual, until=2.0):
    """Return what a visual logs for every frame of a run of a system."""
    recording = RecordingStub()
    simulation = Simulation(system, dt=1.0).prepare().run(until=until)
    problem = simulation.problem_
    visual.set_params(path='cells', recording=recording).prepare(problem)
    trajectory = simulation.trajectory_
    for step, time in enumerate(trajectory.times):
        frame = FrameSpec(step, float(time), trajectory.states[step], simulation.parameters_, problem)
        visual.run(frame)
    return recording.logged


def _image_of(archetype):
    """Return the pixels of an image archetype as rows, columns, and channels."""
    described = archetype.format.as_arrow_array().to_pylist()[0]
    pixels = archetype.buffer.as_arrow_array().to_pylist()[0]
    shape = (described['height'], described['width'], COLOR_CHANNELS)
    return np.asarray(pixels, dtype=np.uint8).reshape(shape)


def test_plaquette_image_and_segments_are_logged_at_every_recorded_frame():
    """Test that a 32 by 32 U(1) run logs 32 by 32 plaquette images and 2048 colored segments at every frame."""
    space = CellComplexSpace(LatticeSpace((SIDE, SIDE))).prepare()
    initial = np.random.default_rng(0).uniform(-np.pi, np.pi, SEGMENTS)
    logged = _logged(_system(space, values={'theta': initial}), CellVisual())
    assert len(logged['cells/theta']) == len(logged['cells/theta/faces']) == 3
    for strips in logged['cells/theta']:
        assert len(strips.strips.as_arrow_array()) == SEGMENTS
        assert len(strips.colors.as_arrow_array()) == SEGMENTS
    for image in logged['cells/theta/faces']:
        assert _image_of(image).shape == (SIDE, SIDE, COLOR_CHANNELS)


def test_recording_of_a_run_with_the_cell_visual_is_saved(tmp_path):
    """Test that a group that names the cell visual runs with a Rerun observer that saves its stream."""
    space = CellComplexSpace(LatticeSpace((SIDE, SIDE))).prepare()
    path = tmp_path / 'run.rrd'
    system = _system(space, visual=VisualSpec('cell'))
    Simulation(system, dt=1.0, observers=[RerunObserver('cells', save_path=path)]).prepare().run(until=2.0)
    assert path.stat().st_size > 0


def test_plaquette_image_uses_a_cyclic_palette():
    """Test that plaquette sums that differ by a period share one color, and that an offset changes the color."""
    space = CellComplexSpace(LatticeSpace((4, 4))).prepare()
    theta = np.zeros(32)
    theta[space.incidences_[1][[0], :].indices[0]] = 2 * np.pi
    image = _image_of(_logged(_system(space, values={'theta': theta}), CellVisual(), until=0.0)['cells/theta/faces'][0])
    assert len(np.unique(image.reshape(-1, COLOR_CHANNELS), axis=0)) == 1
    theta[space.incidences_[1][[0], :].indices[0]] = np.pi
    image = _image_of(_logged(_system(space, values={'theta': theta}), CellVisual(), until=0.0)['cells/theta/faces'][0])
    assert len(np.unique(image.reshape(-1, COLOR_CHANNELS), axis=0)) > 1


def test_three_dimensional_lattice_draws_the_selected_slice():
    """Test that a 3D complex logs the links and the plaquette image of one plane and slice."""
    space = CellComplexSpace(LatticeSpace((3, 4, 5))).prepare()
    logged = _logged(_system(space), CellVisual('theta', plane=(0, 2), section=1), until=0.0)
    assert len(logged['cells/theta'][0].strips.as_arrow_array()) == 2 * 3 * 5
    assert _image_of(logged['cells/theta/faces'][0]).shape == (3, 5, COLOR_CHANNELS)


def test_hyperedge_group_on_a_lattice_logs_an_image():
    """Test that a group on the plaquettes of a lattice complex logs an image of the lattice shape."""
    space = CellComplexSpace(LatticeSpace((4, 3))).prepare()
    logged = _logged(_system(space, ValueElement, 'hyperedges'), CellVisual(), until=0.0)
    assert _image_of(logged['cells/u'][0]).shape == (4, 3, COLOR_CHANNELS)


def test_network_complex_logs_segments_and_polygons():
    """Test that a group on the edges of a network complex logs segments and closed polygons of its faces."""
    graph = nx.cycle_graph(4)
    graph.add_edge(0, 2)
    space = CellComplexSpace(NetworkSpace(graph)).prepare()
    logged = _logged(_system(space, ValueElement, values={'u': np.arange(5.0)}), CellVisual(), until=0.0)
    assert len(logged['cells/u'][0].strips.as_arrow_array()) == 5
    assert len(logged['cells/u/faces'][0].strips.as_arrow_array()) == 2


def test_hyperedge_group_on_a_network_logs_polygons():
    """Test that a group on the faces of a network complex logs closed polygons."""
    space = CellComplexSpace(NetworkSpace(nx.cycle_graph(4)), faces=[(0, 1, 2, 3)]).prepare()
    logged = _logged(_system(space, ValueElement, 'hyperedges'), CellVisual(), until=0.0)
    assert len(logged['cells/u'][0].strips.as_arrow_array()) == 1


def test_views_show_the_state_of_the_group():
    """Test that the visual sets one 2D view of its state."""
    space = CellComplexSpace(LatticeSpace((3, 3))).prepare()
    visual = CellVisual(path='cells').prepare(compile_system(_system(space)))
    assert [type(view).__name__ for view in visual.views_] == ['Spatial2DView']


@pytest.mark.parametrize('plane', [(0, 0), (0, 2)])
def test_prepare_rejects_a_plane_that_is_not_two_axes_of_the_lattice(plane):
    """Test that a plane of a repeated or a missing axis raises an error."""
    space = CellComplexSpace(LatticeSpace((3, 3))).prepare()
    with pytest.raises(ValidationError, match='plane'):
        CellVisual(plane=plane, path='cells').prepare(compile_system(_system(space)))


def test_prepare_rejects_a_group_on_the_sites():
    """Test that a group on the sites of a complex has no cell visual."""
    space = CellComplexSpace(LatticeSpace((3, 3))).prepare()
    system = _system(space, part='sites')
    with pytest.raises(ValidationError, match='edges or hyperedges'):
        CellVisual(path='cells').prepare(compile_system(system))
