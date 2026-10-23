"""Test the `_scenarios` module of `skcomplex.core.experiments`."""

import numpy as np

from skcomplex.core.experiments import (
    Experiment,
    Scenario,
)
from skcomplex.models.economics.resources import harvesting


def test_policy_changes_at_scheduled_time():
    """Test that halving the effort at time 50 changes the stock only after time 50, with the same seeds."""
    scenarios = [Scenario('baseline'), Scenario('quota', schedule=[(50.0, 'harvest.effort', 0.025)])]
    result = Experiment(harvesting, scenarios=scenarios, until=80, outputs=['land/stock.n'], random_state=0).run()
    baseline, quota = result.series('land/stock.n')
    times = result.runs[0].times
    assert np.array_equal(baseline[times <= 50], quota[times <= 50])
    assert quota[-1] > baseline[-1]
    assert result.runs[0].seed == result.runs[1].seed
