"""Test the `_scenarios` module of `skcomplex.experiments`."""

import numpy as np

from skcomplex.experiments import (
    Experiment,
    Scenario,
)
from skcomplex.models.economics.resources import harvesting


def test_policy_changes_at_scheduled_time():
    """Test that halving the effort at time 50 changes the stock only after time 50, with the same seeds."""
    scenarios = [Scenario('baseline'), Scenario('quota', schedule=[(50.0, 'harvest.effort', 0.025)])]
    experiment = Experiment(harvesting, scenarios=scenarios, until=80, outputs=['land/stock.n'], random_state=0)
    result = experiment.prepare().run().result_
    baseline, quota = result.series('land/stock.n')
    times = result.runs[0].times
    assert np.array_equal(baseline[times <= 50], quota[times <= 50])
    assert quota[-1] > baseline[-1]
    assert result.runs[0].seed == result.runs[1].seed


def test_schedule_keeps_the_initial_value_before_its_first_change():
    """Test that a parameter keeps its initial value before its first change and the last due change after it."""
    schedule = [(50.0, 'harvest.effort', 0.025), (20.0, 'harvest.effort', 0.04)]
    scenarios = [Scenario('baseline'), Scenario('late', schedule=schedule)]
    experiment = Experiment(harvesting, scenarios=scenarios, until=80, outputs=['land/stock.n'], random_state=0)
    result = experiment.prepare().run().result_
    baseline, late = result.series('land/stock.n')
    times = result.runs[0].times
    assert np.array_equal(baseline[times <= 20], late[times <= 20])
    assert not np.array_equal(baseline[times > 21], late[times > 21])
