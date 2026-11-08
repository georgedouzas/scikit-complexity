"""Test the `_urban_interactions` module of `skcomplex.systems.social.organizations`."""

import numpy as np
import pytest

from skcomplex.core.analysis import ScalingResult, scaling
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import sweep
from skcomplex.systems.social.organizations import UrbanScalingInteractionsSystem

SIZES = [100, 200, 400, 800]
SEEDS = [0, 1, 2]


def _fit(internal: float) -> ScalingResult:
    """Return the allometric fit of the meetings of cities over a sweep of their size."""
    observables = UrbanScalingInteractionsSystem().get_run_settings()['observables']
    result = sweep(
        lambda **arguments: UrbanScalingInteractionsSystem(**arguments).prepare().system_,
        {'population': SIZES, 'internal': [internal], 'random_state': SEEDS},
        until=float(max(SIZES)),
        outputs=['total_meetings'],
        observables=observables,
        random_state=0,
    )
    meetings = result.series('total_meetings')[:, -1]
    samples = dict(zip(SIZES, meetings.reshape(len(SIZES), len(SEEDS)), strict=True))
    return scaling(samples, kind='allometric', resamples=300)


def test_urban_scaling_interactions_meetings_are_superlinear_with_internal_links():
    """Test that the exponent of the meetings is above 1 at the rate of internal links that this system assumes."""
    fit = _fit(0.002)
    assert fit.interval[0] > 1.0
    assert fit.exponent == pytest.approx(1.14, abs=0.05)


def test_urban_scaling_interactions_meetings_are_linear_without_internal_links():
    """Test that the exponent of the meetings is near 1 without internal links, whose rate makes it superlinear."""
    assert _fit(0.0).exponent == pytest.approx(1.0, abs=0.05)


def test_urban_scaling_interactions_city_stops_growing_at_its_population():
    """Test that a run past the size of the city leaves its residents and links unchanged."""
    system = UrbanScalingInteractionsSystem(30, internal=0.01).prepare().system_
    trajectory = simulate(system, until=40, random_state=0)
    grown, later = trajectory.network(27), trajectory.network(-1)
    assert grown.nodes.size == later.nodes.size == 30
    assert grown.sources.size == later.sources.size


def test_urban_scaling_interactions_meetings_stay_within_the_neighbours():
    """Test that no resident meets more residents than it had neighbours."""
    system = UrbanScalingInteractionsSystem(30).prepare().system_
    trajectory = simulate(system, until=30, random_state=0)
    meetings = np.asarray(trajectory.values('residents.meetings')[-1])
    degree = np.asarray(trajectory.values('residents.degree')[-2])
    assert (meetings <= degree).all()
    assert meetings.sum() > 0


@pytest.mark.parametrize(
    ('settings', 'message'),
    [
        ({'exponent': 2.0}, 'degree exponent'),
        ({'temperature': 0.0}, 'temperature'),
        ({'temperature': 1.0}, 'temperature'),
    ],
)
def test_urban_scaling_interactions_rejects_invalid_settings(settings: dict[str, float], message: str):
    """Test that an exponent not above 2 and a temperature outside the unit interval raise an error."""
    with pytest.raises(ValidationError, match=message):
        UrbanScalingInteractionsSystem(20, **settings).prepare()
