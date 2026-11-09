"""Test the `_scenarios` module of `skcomplex.core.experiments`."""

import pytest

from skcomplex.core.components.specs import ScenarioSpec
from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import compare_scenarios
from skcomplex.systems.biology.epidemiology import SIRSystem


def test_scenarios_override_model_parameters():
    """Test that a scenario that lowers the infection rate lowers the peak of the infected, from the same seeds."""
    scenarios = [ScenarioSpec(), ScenarioSpec('distancing', overrides={'beta': 0.0001})]
    result = compare_scenarios(SIRSystem(), scenarios, replicates=2, until=100, outputs=['I.x'], random_state=0)
    assert [row['scenario'] for row in result.rows] == ['baseline', 'baseline', 'distancing', 'distancing']
    baseline, _, distancing, _ = result.series('I.x')
    assert distancing.max() < baseline.max()
    assert result.runs[0].seed == result.runs[2].seed


def test_scenarios_must_be_scenario_specs():
    """Test that no scenario, or an object that is no scenario, raises an error."""
    with pytest.raises(ValidationError, match='nonempty sequence of ScenarioSpec'):
        compare_scenarios(SIRSystem(), [])
    with pytest.raises(ValidationError, match='nonempty sequence of ScenarioSpec'):
        compare_scenarios(SIRSystem(), ['baseline'])
