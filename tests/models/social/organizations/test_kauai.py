"""Test the `_kauai` module of `skcomplex.models.social.organizations`."""

import numpy as np
import pytest
from shapely import Point

from skcomplex.errors import ValidationError
from skcomplex.models.social.organizations import kauai_population, load_island_layers, load_island_populations

from ..conftest import _run


def test_island_layers_load_inside_their_boundary():
    """Test that the layers of the synthetic island load from the package and lie inside its boundary."""
    layers = load_island_layers()
    assert load_island_populations() == [100, 50, 0]
    assert all(layers['boundary'][0].contains(polygon) for polygon in layers['census'] + layers['places'])
    assert all(layers['boundary'][0].contains(polygon) for polygon in layers['facilities'])


def test_kauai_population_follows_the_census_layer():
    """Test that populations of 100, 50, and 0 give 150 residents, each with a home point in its census region."""
    census = load_island_layers()['census']
    trajectory = _run(kauai_population((100, 50, 0)), 4)
    start = trajectory.agents('residents', 0)
    regions = start.values['census'].astype(int).tolist()
    assert len(start.ids) == 150
    assert regions == [0] * 100 + [1] * 50
    assert all(census[region].contains(Point(point)) for region, point in zip(regions, start.locations, strict=True))
    homes = trajectory.agents('residents', 4).values['home']
    np.testing.assert_array_equal(homes, start.locations)
    np.testing.assert_array_equal(trajectory.agents('residents', 3).locations, start.locations)


def test_kauai_population_rejects_populations_of_other_regions():
    """Test that populations that do not match the census layer raise an error that names both."""
    with pytest.raises(ValidationError, match='3 census regions'):
        kauai_population((100, 50))


def test_kauai_residents_commute_through_their_regions():
    """Test that residents stand in their facility, then in their favourite place, and count the visits."""
    layers = load_island_layers()
    trajectory = _run(kauai_population((10, 5, 0)), 6)
    at_work = trajectory.agents('residents', 1)
    at_place = trajectory.agents('residents', 2)
    assert all(
        layers['facilities'][int(work)].contains(Point(point))
        for work, point in zip(at_work.values['work'], at_work.locations, strict=True)
    )
    assert all(
        layers['places'][int(place)].contains(Point(point))
        for place, point in zip(at_place.values['favourite'], at_place.locations, strict=True)
    )
    assert trajectory.agents('residents', 6).values['visits'].tolist() == [2.0] * 15


def test_kauai_positive_event_raises_sentiment_of_those_who_weigh_its_topic():
    """Test that a positive event raises the mean of the residents who weigh it, and the others follow by influence."""
    alone = _run(kauai_population((100, 50, 0), events=[(1, 0, 0.5)], influence=0.0), 3)
    start, end = alone.agents('residents', 0), alone.agents('residents', 3)
    weighted = start.values['weights'][:, 0] > 0
    assert weighted.sum() == 75
    assert end.values['sentiment'][weighted].mean() > start.values['sentiment'][weighted].mean() + 0.2
    np.testing.assert_array_equal(end.values['sentiment'][~weighted], start.values['sentiment'][~weighted])
    together = _run(kauai_population((100, 50, 0), events=[(1, 0, 0.5)], influence=0.2), 6)
    others = together.agents('residents', 6).values['sentiment'][~weighted]
    assert others.mean() > start.values['sentiment'][~weighted].mean()


def test_kauai_sentiment_stays_between_the_bounds():
    """Test that a strong event leaves every sentiment within -1 and 1."""
    trajectory = _run(kauai_population((20, 10, 0), events=[(1, 0, 5.0), (2, 1, -5.0)]), 3)
    for frame in (1, 2, 3):
        sentiment = trajectory.agents('residents', frame).values['sentiment']
        assert sentiment.min() >= -1.0
        assert sentiment.max() <= 1.0
    assert trajectory.agents('residents', 1).values['sentiment'].max() == 1.0
