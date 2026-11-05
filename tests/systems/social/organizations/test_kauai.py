"""Test the `_kauai` module of `skcomplex.systems.social.organizations`."""

import numpy as np
import pytest
from shapely import Point

from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.errors import ValidationError
from skcomplex.systems.social.organizations import (
    ResidentElement,
    SentimentInteraction,
    kauai_population,
    load_island_layers,
    load_island_populations,
)

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


def _pair(first, second, **parameters):
    """Return a system of two residents at one point who meet, with a sentiment each and no topic in common."""
    values = {'sentiment': np.array([first, second]), 'weights': np.array([[1.0, 0.0], [0.0, 1.0]]), **parameters}
    residents = GroupElement(
        'residents',
        element=ResidentElement,
        size=2,
        values=values,
        location=[(500.0, 500.0), (500.0, 500.0)],
    )
    return System(
        elements=[residents],
        interactions=[SentimentInteraction('sentiment', target='residents')],
        space=ContinuousSpace(extent=(1000.0, 1000.0)),
    )


def test_kauai_event_follows_equation_five_of_the_chapter():
    """Test the terms of the example of Eq. 6, which the chapter adds to -0.2755 by a slip of sign on its last term."""
    residents = GroupElement(
        'residents',
        element=ResidentElement,
        size=1,
        values={'sentiment': np.array([-0.3]), 'weights': np.array([[0.5, 0.1, 0.0, 0.2]]), 'event_rate': 0.05},
        location=[(500.0, 500.0)],
    )
    event = [(1, 0, 0.8), (1, 1, -0.1), (1, 2, 0.4), (1, 3, 0.1)]
    system = System(
        elements=[residents],
        interactions=[SentimentInteraction('sentiment', target='residents', events=event)],
        space=ContinuousSpace(extent=(1000.0, 1000.0)),
    )
    sentiment = _run(system, 1).agents('residents', 1).values['sentiment']
    assert sentiment[0] == pytest.approx(-0.3 + 0.0275 + 0.001 + 0.0 + 0.004)


def test_kauai_partners_converge_by_equation_four_of_the_chapter():
    """Test that two residents who start one interaction each change by `a + (1 - |a| CR) (b - a)` at each of them."""
    convergence = 0.5
    end = _run(_pair(0.5, -0.5, convergence=convergence), 1).agents('residents', 1).values['sentiment']
    expected = []
    for order in ((0, 1), (1, 0)):
        values = [0.5, -0.5]
        for one in order:
            other = 1 - one
            mu_one, mu_other = 1 - abs(values[one]) * convergence, 1 - abs(values[other]) * convergence
            changes = mu_one * (values[other] - values[one]), mu_other * (values[one] - values[other])
            values[one], values[other] = values[one] + changes[0], values[other] + changes[1]
        expected.append(values)
    assert any(np.allclose(end, values) for values in expected)


def test_kauai_strong_opinion_moves_less_than_an_undecided_one():
    """Test that with `CR` of 1 a resident of sentiment 1 keeps it, and the undecided one takes it."""
    end = _run(_pair(1.0, 0.0, convergence=1.0), 1).agents('residents', 1).values['sentiment']
    np.testing.assert_allclose(end, [1.0, 1.0])


def test_kauai_interaction_stops_at_the_grid_cell():
    """Test that residents of different grid cells keep their sentiments, though they stand close."""
    residents = GroupElement(
        'residents',
        element=ResidentElement,
        size=2,
        values={'sentiment': np.array([0.5, -0.5])},
        location=[(990.0, 500.0), (1010.0, 500.0)],
    )
    system = System(
        elements=[residents],
        interactions=[SentimentInteraction('sentiment', target='residents')],
        space=ContinuousSpace(extent=(2000.0, 1000.0)),
    )
    np.testing.assert_array_equal(_run(system, 3).agents('residents', 3).values['sentiment'], [0.5, -0.5])


def test_kauai_positive_event_raises_sentiment_of_those_who_weigh_its_topic():
    """Test that a positive event raises the mean of those who weigh it, and the others follow by interaction."""
    alone = _run(kauai_population((100, 50, 0), events=[(1, 0, 0.8)], event_rate=0.5, partners=0), 3)
    start, end = alone.agents('residents', 0), alone.agents('residents', 3)
    weighted = start.values['weights'][:, 0] > 0
    assert weighted.sum() == 75
    assert end.values['sentiment'][weighted].mean() > start.values['sentiment'][weighted].mean() + 0.2
    np.testing.assert_array_equal(end.values['sentiment'][~weighted], start.values['sentiment'][~weighted])
    together = _run(kauai_population((100, 50, 0), events=[(1, 0, 0.8)], event_rate=0.5), 6)
    others = together.agents('residents', 6).values['sentiment'][~weighted]
    assert others.mean() > start.values['sentiment'][~weighted].mean()


def test_kauai_sentiment_stays_between_the_bounds():
    """Test that strong events and interactions leave every sentiment within -1 and 1."""
    trajectory = _run(kauai_population((20, 10, 0), events=[(1, 0, 1.0), (2, 1, -1.0)], event_rate=1.0), 3)
    for frame in (1, 2, 3):
        sentiment = trajectory.agents('residents', frame).values['sentiment']
        assert sentiment.min() >= -1.0
        assert sentiment.max() <= 1.0
