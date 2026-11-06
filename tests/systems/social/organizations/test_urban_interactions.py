"""Test the `_urban_interactions` module of `skcomplex.systems.social.organizations`."""

import numpy as np
import pytest

from skcomplex.core.analysis import ScalingAnalysis
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import Experiment, GridDesign
from skcomplex.systems import describe
from skcomplex.systems.social.organizations import urban_scaling_interactions

SIZES = [100, 200, 400, 800]
SEEDS = [0, 1, 2]


def _fit(internal: float) -> ScalingAnalysis:
    """Return the allometric fit of the meetings of cities over a sweep of their size."""
    observables = describe('urban_scaling_interactions').observables_of(urban_scaling_interactions(20))
    experiment = Experiment(
        urban_scaling_interactions,
        GridDesign(),
        {'population': SIZES, 'internal': [internal], 'random_state': SEEDS},
        until=float(max(SIZES)),
        outputs=['total_meetings'],
        observables=observables,
        random_state=0,
    )
    meetings = experiment.prepare().run().result_.series('total_meetings')[:, -1]
    samples = dict(zip(SIZES, meetings.reshape(len(SIZES), len(SEEDS)), strict=True))
    return ScalingAnalysis('allometric', resamples=300).prepare(samples).run()


def test_urban_scaling_interactions_meetings_are_superlinear_with_internal_links():
    """Test that the exponent of the meetings is above 1 at the rate of internal links that this system assumes."""
    fit = _fit(0.002)
    assert fit.interval_[0] > 1.0
    assert fit.exponent_ == pytest.approx(1.14, abs=0.05)


def test_urban_scaling_interactions_meetings_are_linear_without_internal_links():
    """Test that the exponent of the meetings is near 1 without internal links, whose rate makes it superlinear."""
    assert _fit(0.0).exponent_ == pytest.approx(1.0, abs=0.05)


def test_urban_scaling_interactions_city_stops_growing_at_its_population():
    """Test that a run past the size of the city leaves its residents and links unchanged."""
    system = urban_scaling_interactions(30, internal=0.01)
    trajectory = Simulation(system, random_state=0).prepare().run(until=40).trajectory_
    grown, later = trajectory.network(27), trajectory.network(-1)
    assert grown.nodes.size == later.nodes.size == 30
    assert grown.sources.size == later.sources.size


def test_urban_scaling_interactions_meetings_stay_within_the_neighbours():
    """Test that no resident meets more residents than it had neighbours."""
    system = urban_scaling_interactions(30)
    trajectory = Simulation(system, random_state=0).prepare().run(until=30).trajectory_
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
        urban_scaling_interactions(20, **settings)
