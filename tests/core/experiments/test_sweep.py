"""Test the `_sweep` module of `skcomplex.core.experiments`."""

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.experiments import sweep
from skcomplex.core.systems import System
from skcomplex.systems.biology.ecology import LogisticGrowthSystem
from skcomplex.systems.biology.epidemiology import SIRSystem
from skcomplex.systems.mathematics.dynamical_systems import HenonMapInteraction, HenonMapSystem


def _logistic(s):
    """Return the logistic map `x -> r x (1 - x)`."""
    x = s.target.x
    return [UpdateLaw(x, s.target.r * x * (1 - x))]


def logistic_map():
    """Return the logistic map of the element `map`, with `x = 0.2` and `r = 3`."""
    logistic = Interaction('logistic', roles={'target': 'map'}, laws=_logistic)
    return System(elements=[Element('map', values={'x': 0.2, 'r': 3.0})], interactions=[logistic])


def henon():
    """Return the Hénon map as a `System` of one element and one interaction."""
    return System(
        'henon_map',
        elements=[Element('map', values={'x': 0.0, 'y': 0.0})],
        interactions=[HenonMapInteraction('henon', target='map')],
    )


def count_prepares(monkeypatch):
    """Return the list that records the label of every system `System._prepare` compiles."""
    calls = []
    prepare = System._prepare

    def counted(self, context=None):
        calls.append(self.label)
        prepare(self, context)

    monkeypatch.setattr(System, '_prepare', counted)
    return calls


def last_value(trajectory):
    """Return the last infected count."""
    return float(trajectory.get_values('I.x')[-1])


def test_sweep_of_one_parameter_gives_one_row_per_value():
    """Test that a sweep over one model parameter gives one tidy row per value with its output."""
    result = sweep(LogisticGrowthSystem(), {'K': [50.0, 100.0]}, until=40, outputs=['population.x'])
    assert result.get_column('K').tolist() == [50.0, 100.0]
    np.testing.assert_allclose(result.get_column('population.x'), [50.0, 100.0], rtol=1e-3)
    assert list(result.convert_to_frame().columns[:4]) == ['K', 'scenario', 'replicate', 'seed']


def test_orbit_diagram_sweep():
    """Test that a sweep of 200 growth rates of the logistic map keeps the last 100 values of each run."""
    system = logistic_map()
    rates = np.linspace(2.8, 4.0, 200)
    result = sweep(system, {'map.r': rates}, until=400, warmup=301, outputs=['map.x'])
    assert len(result.rows) == 200
    assert result.get_column('map.r').tolist() == rates.tolist()
    assert result.get_series('map.x').shape == (200, 100)


def test_sampled_sweep_stays_within_bounds():
    """Test that a sampled sweep runs `n` rows within the bounds."""
    result = sweep(LogisticGrowthSystem(), {'r': (0.5, 1.0)}, design='sobol', n=4, until=1, random_state=0)
    assert len(result.rows) == 4
    assert all(0.5 <= value <= 1.0 for value in result.get_column('r'))


def test_summary_across_replicates():
    """Test that the summary gives the mean, standard deviation, and quantiles per row."""
    result = sweep(
        SIRSystem(stochastic=True),
        {'gamma': [0.1, 0.2]},
        replicates=5,
        until=30,
        outputs=[last_value],
        random_state=1,
    )
    summary = result.get_summary()
    assert len(summary) == 2
    for row in summary:
        assert row['runs'] == 5
        assert {'last_value.mean', 'last_value.std', 'last_value.q0.05', 'last_value.q0.5', 'last_value.q0.95'} <= set(
            row,
        )
    assert result.convert_to_frame().shape[0] == 10


def test_sweep_leaves_the_system_of_the_user_unchanged():
    """Test that a sweep of a nested parameter runs clones, so the system keeps its parameters and is not prepared."""
    system = henon()
    before = system.get_params()
    result = sweep(system, {'henon__a': [1.2, 1.4]}, until=20, outputs=['map.x'])
    assert system.get_params() == before
    assert not hasattr(system, 'system_')
    assert result.get_column('henon__a').tolist() == [1.2, 1.4]


def test_sweep_of_nested_parameters_matches_the_ready_made_system():
    """Test that the nested parameter `henon__a` of a `System` gives the runs of `HenonMapSystem(a=...)`."""
    result = sweep(henon(), {'henon__a': [1.2, 1.4]}, until=20, outputs=['map.x'])
    for a, series in zip([1.2, 1.4], result.get_series('map.x'), strict=True):
        expected = HenonMapSystem(a=a).prepare().simulate(20).get_values('map.x')
        np.testing.assert_array_equal(series, expected)


def test_sweep_prepares_per_parameter_and_once_for_initial_states(monkeypatch):
    """Test that a sweep prepares once per value of a nested parameter and once for many initial states."""
    calls = count_prepares(monkeypatch)
    sweep(henon(), {'henon__a': [1.1, 1.2, 1.3]}, until=5, outputs=['map.x'])
    assert len(calls) == 3
    calls.clear()
    sweep(henon(), {'map.x': [0.0, 0.1, 0.2], 'henon.b': [0.2, 0.3]}, until=5, outputs=['map.x'])
    assert len(calls) == 1


def test_sweep_of_a_ready_made_system_by_model_parameter_and_path():
    """Test that a model parameter and the dotted path of the same parameter give the same runs."""
    by_parameter = sweep(HenonMapSystem(), {'a': [1.2, 1.4]}, until=20, outputs=['map.x'])
    by_path = sweep(HenonMapSystem(), {'henon.a': [1.2, 1.4]}, until=20, outputs=['map.x'])
    np.testing.assert_allclose(by_parameter.get_series('map.x'), by_path.get_series('map.x'), atol=1e-12)
