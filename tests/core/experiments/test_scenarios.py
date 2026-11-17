"""Test the `_scenarios` module of `skcomplex.core.experiments`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import ScenarioSpec, compare_scenarios
from skcomplex.core.systems import System
from skcomplex.systems.biology.epidemiology import SIRSystem
from skcomplex.systems.mathematics.dynamical_systems import HenonMapInteraction


def test_scenarios_override_model_parameters():
    """Test that a scenario that lowers the infection rate lowers the peak of the infected, from the same seeds."""
    scenarios = [ScenarioSpec(), ScenarioSpec('distancing', overrides={'beta': 0.0001})]
    result = compare_scenarios(SIRSystem(), scenarios, replicates=2, until=100, outputs=['I.x'], random_state=0)
    assert [row['scenario'] for row in result.rows] == ['baseline', 'baseline', 'distancing', 'distancing']
    baseline, _, distancing, _ = result.get_series('I.x')
    assert distancing.max() < baseline.max()
    assert result.runs[0].seed == result.runs[2].seed


def test_scenarios_must_be_scenario_specs():
    """Test that no scenario, or an object that is no scenario, raises an error."""
    with pytest.raises(ValidationError, match='nonempty sequence of ScenarioSpec'):
        compare_scenarios(SIRSystem(), [])
    with pytest.raises(ValidationError, match='nonempty sequence of ScenarioSpec'):
        compare_scenarios(SIRSystem(), ['baseline'])


def _henon():
    """Return the Hénon map as a `System` of one element and one interaction."""
    return System(
        'henon_map',
        elements=[Element('map', values={'x': 0.0, 'y': 0.0})],
        interactions=[HenonMapInteraction('henon', target='map')],
    )


def test_scenarios_of_a_system_set_nested_parameters_and_schedules():
    """Test that a nested parameter set before the run and a dotted parameter set from the start give the same run."""
    scenarios = [
        ScenarioSpec(),
        ScenarioSpec('weaker', overrides={'henon__a': 1.2}),
        ScenarioSpec('switched', schedule=[(0.0, 'henon.a', 1.2)]),
    ]
    result = compare_scenarios(_henon(), scenarios, until=10, outputs=['map.x'])
    baseline, weaker, switched = result.get_series('map.x')
    np.testing.assert_allclose(weaker, switched, atol=1e-12)
    assert not np.allclose(baseline, weaker)
