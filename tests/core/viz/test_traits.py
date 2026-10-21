"""Test the visual of the traits of agents without a space."""

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.compile import compile_system
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import AgentSnapshot
from skcomplex.core.viz import AgentsVisual, RerunObserver, TraitsVisual, agents_visual_of, default_blueprint
from skcomplex.models import create


class Logs:
    """Recording stand-in that keeps the entity paths and archetypes it receives."""

    def __init__(self):
        """Start with no logs."""
        self.logs = []

    def log(self, path, archetype, static=False):
        """Keep a logged entity path and archetype, leaving out static styles."""
        if not static:
            self.logs.append((path, archetype))

    def of(self, path):
        """Return the archetypes logged under a path."""
        return [archetype for logged, archetype in self.logs if logged == path]


def _views(blueprint):
    """Return the views of a blueprint in layout order."""
    found = []
    stack = [blueprint.root_container]
    while stack:
        part = stack.pop(0)
        if isinstance(part, rrb.View):
            found.append(part)
        else:
            stack = list(part.contents) + stack
    return found


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


def test_default_blueprint_of_moran_process_shows_trait_statistics_and_histogram(monkeypatch, tmp_path):
    """Test that the Moran process shows its trait over time and its histogram, and logs the mean of every trait."""
    problem = compile_system(create('moran_process'))
    views = [type(view).__name__ for view in _views(default_blueprint(problem))]
    assert views[:2] == ['TimeSeriesView', 'BarChartView']
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


def test_traits_visual_counts_distinct_integer_values():
    """Test that an integer trait such as a genome label logs its number of distinct values."""
    visual, logs = TraitsVisual(), Logs()
    snapshot = AgentSnapshot(np.arange(5), {'trait': np.array([3.0, 3.0, 17.0, 200.0, 17.0])})
    visual.log_agents(logs, 'organisms', snapshot, None)
    assert logs.of('organisms/trait/distinct')[0].scalars.as_arrow_array().to_pylist() == [3.0]


def test_traits_visual_logs_each_agent_of_small_groups():
    """Test that a group of at most ten agents logs every agent's trait as its own series, and larger ones do not."""
    visual, logs = TraitsVisual(), Logs()
    visual.log_agents(logs, 'trials', AgentSnapshot(np.arange(3), {'successes': np.array([1.0, 2.0, 0.0])}), None)
    assert logs.of('trials/successes/agents')[0].scalars.as_arrow_array().to_pylist() == [1.0, 2.0, 0.0]
    visual.log_agents(logs, 'crowd', AgentSnapshot(np.arange(11), {'successes': np.ones(11)}), None)
    assert not logs.of('crowd/successes/agents')


def test_traits_visual_shows_chosen_statistics_first():
    """Test that the first view shows the chosen statistics of the colored trait."""
    problem = compile_system(create('tangled_nature', organisms=20))
    view = TraitsVisual(color_by='trait', shown=('distinct',)).views(problem.agents[0], problem)[0]
    assert view.contents == ['+ /organisms/trait/distinct']


def test_traits_visual_can_show_the_histogram_first():
    """Test that a traits visual can put the histogram of its trait before its time series."""
    problem = compile_system(create('deffuant'))
    views = TraitsVisual(color_by='opinion', histogram_first=True).views(problem.agents[0], problem)
    assert [type(view).__name__ for view in views] == ['BarChartView', 'TimeSeriesView', 'TimeSeriesView']


def test_traits_visual_totals_rows_of_numbers():
    """Test that a trait holding a row of numbers per agent, such as an inventory of known words, logs its total."""
    visual, logs = TraitsVisual(), Logs()
    words = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    visual.log_agents(logs, 'namers', AgentSnapshot(np.arange(3), {'words': words}), None)
    assert logs.of('namers/words/mean')[0].scalars.as_arrow_array().to_pylist() == [2.0]


def test_agents_visual_draws_one_dimensional_positions_on_a_line():
    """Test that agents whose position state has one coordinate are drawn as points on a line in a 2D view."""
    visual, logs = AgentsVisual(position='x', color_by='v'), Logs()
    snapshot = AgentSnapshot(np.arange(3), {'x': np.array([1.0, 4.0, 9.0]), 'v': np.array([0.0, 1.0, 2.0])})
    visual.log_agents(logs, 'cars', snapshot, None)
    points = logs.of('cars')[0].positions.as_arrow_array().to_pylist()
    assert points == [[1.0, 0.0], [4.0, 0.0], [9.0, 0.0]]


def test_charbonneau_traffic_has_a_road_view():
    """Test that the cars of Charbonneau's traffic model keep the agents visual and get a 2D view of the road."""
    problem = compile_system(create('charbonneau_traffic'))
    visual = agents_visual_of(problem.agents[0], problem)
    assert not isinstance(visual, TraitsVisual)
    views = visual.views(problem.agents[0], problem)
    assert type(views[0]).__name__ == 'Spatial2DView'
