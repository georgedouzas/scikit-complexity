"""Test the `_avalanches` module of `skcomplex.models.physics.criticality`."""

import numpy as np
import pytest

from skcomplex.analysis import EventAnalysis, FitAnalysis
from skcomplex.engine.runtime import Simulation
from skcomplex.models import create
from skcomplex.models.physics.criticality import AvalancheAnalysis


def _exponent(values: np.ndarray, minimum: float) -> FitAnalysis:
    """Return the discrete power law fitted above a fixed lower cutoff."""
    return FitAnalysis('power', minimum=minimum, discrete=True, bootstrap=0).check().prepare(values).run()


def test_avalanches_equal_the_excursions():
    """Test that the avalanches of an activity series equal its excursions above the threshold."""
    activity = np.random.default_rng(0).poisson(2, 500) * (np.random.default_rng(1).random(500) < 0.5)
    found = AvalancheAnalysis(threshold=0.5).check().prepare(activity).run()
    excursions = EventAnalysis('excursions', threshold=0.5).check().prepare(activity).run()
    np.testing.assert_array_equal(found.sizes_, excursions.sizes_)
    np.testing.assert_array_equal(found.durations_, excursions.durations_)
    assert len(found.shapes_) == len(excursions.shapes_)


@pytest.mark.slow
@pytest.mark.parametrize(
    ('name', 'parameters', 'until', 'transient', 'bounds'),
    [
        ('sandpile', {'shape': (96, 96)}, 20_000, 12_000, (1.15, 1.35)),
        ('earthquakes', {'shape': (32, 32), 'alpha': 0.2}, 30_000, 5_000, (1.7, 2.1)),
    ],
)
def test_avalanches_same_statistics_on_a_sandpile_and_an_earthquake_model(name, parameters, until, transient, bounds):
    """Test the size exponents of a Bak-Tang-Wiesenfeld sandpile and an Olami-Feder-Christensen model."""
    trajectory = Simulation(create(name, **parameters), dt=1000).run(until=until).trajectory_
    records = [record for record in trajectory.events if record.time > transient and record.size > 0]
    found = AvalancheAnalysis().check().prepare(records).run()
    size = _exponent(found.sizes_, 10)
    low, high = bounds
    assert low <= size.exponent_ <= high
    ratio, p_value = size.comparisons_['exponential']
    assert ratio > 0
    assert p_value < 0.05
