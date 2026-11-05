"""Test the `_linear_programs` module of `skcomplex.engine.viz`."""

from pathlib import Path

import numpy as np
import pytest
import rerun as rr

from skcomplex.engine.compile import compile_system
from skcomplex.engine.problems import FrameSpec
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import LinearProgramVisual, RerunObserver
from skcomplex.errors import ValidationError
from skcomplex.models import create

TOY_VARIABLES = 10
TOY_ROWS = 8


def _prepared(**settings):
    """Return the problem of the toy network and its visual, prepared with a recording."""
    problem = compile_system(create('flux_balance'))
    recording = rr.RecordingStream('linear_program')
    visual = LinearProgramVisual(path='cell', recording=recording, **settings)
    return problem, visual.check().prepare(problem)


def _logged(monkeypatch, visual, problem):
    """Return what one frame of the program logs, by entity path."""
    logged = {}

    def log(self, path, *entries, **options):
        logged.setdefault(path, []).extend(entries)
        del self, options

    monkeypatch.setattr(rr.RecordingStream, 'log', log)
    solved = problem.algebraic(0.0, problem.initial_state, problem.initial_parameters)
    visual.run(FrameSpec(0, 0.0, solved, problem.initial_parameters, problem))
    return logged


def test_visual_logs_bars_in_declared_order_and_the_objective(monkeypatch):
    """Test that a frame logs a bar chart of the variables in order, and the objective as a scalar."""
    problem, visual = _prepared()
    logged = _logged(monkeypatch, visual, problem)
    bars = logged['cell/variables'][0]
    values = np.asarray(bars.values.as_arrow_array().to_pylist()[0]['buffer'])
    assert values.size == TOY_VARIABLES
    assert values[-1] == pytest.approx(50)
    assert 'cell/objective' in logged


def test_visual_lays_the_graph_out_with_a_seed(monkeypatch):
    """Test that two visuals with one seed place one node per balance and per variable at the same positions."""
    _, first = _prepared(graph=True, seed=3)
    problem, second = _prepared(graph=True, seed=3)
    assert set(first.positions_) == set(second.positions_)
    assert len(first.positions_) == TOY_VARIABLES + TOY_ROWS
    for name, position in first.positions_.items():
        np.testing.assert_array_equal(position, second.positions_[name])
    logged = _logged(monkeypatch, second, problem)
    assert 'cell/graph' in logged
    strips = logged['cell/flows'][0]
    widths = np.asarray(strips.radii.as_arrow_array().to_pylist(), dtype=float)
    assert len(set(widths.round(6).tolist())) > 1


def test_visual_leaves_the_graph_out_on_request():
    """Test that a visual without a graph holds no positions and two views."""
    _, visual = _prepared(graph=False)
    assert visual.positions_ == {}
    assert len(visual.views_) == 2


def test_visual_rejects_an_element_without_a_program():
    """Test that an element that no program sets raises an error."""
    problem = compile_system(create('lotka_volterra'))
    with pytest.raises(ValidationError, match='no program sets a state of prey'):
        LinearProgramVisual(path='prey', recording=rr.RecordingStream('none')).check().prepare(problem)


def test_observer_streams_a_batch_culture_to_a_file(tmp_path: Path):
    """Test that a dynamic FBA run saves its recording with the variables and the objective series."""
    observer = RerunObserver('culture', save_path=tmp_path / 'run.rrd', blueprint='default', interval=10)
    simulation = Simulation(create('dynamic_fba'), dt=0.05, observers=[observer])
    trajectory = simulation.check().prepare().run(until=1.0).trajectory_
    assert (tmp_path / 'run.rrd').exists()
    assert observer.logged_frames_ == 3
    objective = trajectory.values('fba/cell.objective_0')
    assert objective[-1] == pytest.approx(0.8739, rel=0.01)


def test_observer_logs_the_growth_of_every_site_as_an_image(tmp_path: Path):
    """Test that a lattice logs its growth variable through the image visual of its group."""
    observer = RerunObserver('colony', save_path=tmp_path / 'colony.rrd', interval=1)
    system = create('comets', shape=(4, 4))
    simulation = Simulation(system, dt=0.1, observers=[observer])
    simulation.check().prepare().run(until=0.1)
    assert (tmp_path / 'colony.rrd').exists()
