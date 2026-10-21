"""Test the visual of the traits of agents without a space."""

import numpy as np
import rerun as rr

from skcomplex.core.compile import compile_system
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import AgentSnapshot
from skcomplex.core.viz import RerunObserver, TraitsVisual, agents_visual_of, default_blueprint
from skcomplex.models import create


class Logs:
    """Recording stand-in that keeps the entity paths and archetypes it receives."""

    def __init__(self):
        """Start with no logs."""
        self.logs = []

    def log(self, path, archetype, static=False):
        """Keep a logged entity path and archetype."""
        self.logs.append((path, archetype))

    def of(self, path):
        """Return the archetypes logged under a path."""
        return [archetype for logged, archetype in self.logs if logged == path]


def _views(view):
    """Return every view below a container view, depth first."""
    contents = getattr(view, 'contents', None)
    if isinstance(contents, list) and contents and not isinstance(contents[0], str):
        return [item for child in contents for item in _views(child)]
    return [view]


def test_traits_visual_is_the_default_of_agents_without_space():
    """Test that agents without a space take the traits visual, and agents in a space keep the agents visual."""
    problem = compile_system(create('moran_process'))
    assert isinstance(agents_visual_of(problem.agents[0], problem), TraitsVisual)
    problem = compile_system(create('bonabeau_hierarchies', random_state=0))
    assert not isinstance(agents_visual_of(problem.agents[0], problem), TraitsVisual)


def test_traits_visual_logs_histogram_and_statistics():
    """Test that a numeric trait logs a histogram, its mean, and its standard deviation."""
    visual, logs = TraitsVisual(color_by='fitness'), Logs()
    snapshot = AgentSnapshot(np.arange(4), {'fitness': np.array([1.0, 1.0, 2.0, 4.0])})
    visual.log_agents(logs, 'population', snapshot, None)
    histogram = logs.of('population/fitness/histogram')[0]
    assert type(histogram).__name__ == 'BarChart'
    assert sum(histogram.values.as_arrow_array()[0].as_py()['buffer']) == 4
    assert logs.of('population/fitness/mean')[0].scalars.as_arrow_array().to_pylist() == [2.0]


def test_traits_visual_counts_collections():
    """Test that a trait holding collections, such as the vocabularies of the naming game, logs their sizes."""
    visual, logs = TraitsVisual(), Logs()
    words = np.empty(3, dtype=object)
    words[:] = [['a'], ['a', 'b'], ['a', 'b', 'c']]
    visual.log_agents(logs, 'namers', AgentSnapshot(np.arange(3), {'words': words}), None)
    assert logs.of('namers/words/mean')[0].scalars.as_arrow_array().to_pylist() == [2.0]


def test_default_blueprint_of_moran_process_shows_a_trait_histogram(monkeypatch, tmp_path):
    """Test that the first agent view of the Moran process is a bar chart and that every numeric trait has a mean."""
    problem = compile_system(create('moran_process'))
    views = [item for view in default_blueprint(problem).root_container.contents for item in _views(view)]
    assert any(type(view).__name__ == 'BarChartView' for view in views)
    logged = []
    original = rr.RecordingStream.log

    def log(self, path, *args, **kwargs):
        logged.append(path)
        return original(self, path, *args, **kwargs)

    monkeypatch.setattr(rr.RecordingStream, 'log', log)
    observer = RerunObserver('moran', save_path=tmp_path / 'moran.rrd')
    Simulation(create('moran_process'), observers=[observer], random_state=0).run(until=10)
    assert 'population/trait/mean' in logged
    assert 'population/fitness/mean' in logged
