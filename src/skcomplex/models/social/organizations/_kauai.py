"""Kauai population model: residents placed by census regions, whose sentiment follows events and the people around."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence

import numpy as np
import shapely
from shapely import Polygon

from ....core.elements import GroupElement
from ....core.spaces import ContinuousSpace
from ....core.system import System
from ....errors import ValidationError
from ....specs import ObservableSpec
from ..._catalog import register_model
from ._commute import CommuteInteraction
from ._island import load_island_layers, load_island_populations
from ._place_visits import PlaceVisitsInteraction
from ._resident import ResidentElement
from ._sentiment import SentimentInteraction

MIN_WEIGHT = 0.5


def _get_weights(n_residents: int, n_topics: int, share: float, rng: np.random.Generator) -> np.ndarray:
    """Return the weight of every topic per resident, above zero for the given share of residents of every topic."""
    weights = np.zeros((n_residents, n_topics))
    n_weighted = round(share * n_residents)
    for topic in range(n_topics):
        weighted = rng.permutation(n_residents)[:n_weighted]
        weights[weighted, topic] = rng.uniform(MIN_WEIGHT, 1.0, n_weighted)
    return weights


def _mean_sentiment(system: System) -> list[ObservableSpec]:
    """Return the mean sentiment of the residents."""
    return [ObservableSpec('mean_sentiment', group='residents', variable='sentiment', reduce='mean')]


@register_model(
    field='social/organizations',
    tags=('agents', 'continuous space', 'region layers', 'census', 'sentiment'),
    references=('Carmichael et al. 2019',),
    defaults={'until': 30},
    observables=_mean_sentiment,
)
def kauai_population(
    populations: Sequence[int] | None = None,
    *,
    layers: Mapping[str, Sequence[Polygon]] | None = None,
    n_topics: int = 2,
    weighted_share: float = 0.5,
    events: Sequence[tuple[int, int, float]] = ((1, 0, 0.2),),
    influence: float = 0.1,
    radius: float = 2000.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return one agent per resident of the census regions of a map, moving between home, work, and favourite places.

    Every resident starts at a uniform point of its census region, its home, and has a facility of work and a favourite
    place drawn uniformly. It commutes between the three in a cycle of three steps. Its sentiment rises and falls with
    the events on the topics it weighs and moves towards the mean sentiment of the residents within `radius`, in the
    manner of the model of Zanbaka et al. in Carmichael et al. (2019 ch. 3). The default map is a synthetic island.

    Args:
        populations:
            The number of residents of every census region, or `None` for the populations of the synthetic island.

        layers:
            The polygons of the layers `boundary`, `census`, `places`, and `facilities` in non-negative metric
            coordinates, or `None` for the synthetic island.

        n_topics:
            The number of topics.

        weighted_share:
            The share of residents that weight a topic above zero.

        events:
            The events as `(time, topic, strength)`.

        influence:
            The share of the difference from the mean sentiment of the residents around that one step closes.

        radius:
            The distance within which a resident hears others, in the units of the coordinates.

        random_state:
            The seed or generator of the initial sentiments, the weights, the work, and the favourite places.

    Returns:
        system:
            The model, with the group `residents`, its states `sentiment`, `weights`, `home`, `census`, `work`,
            `favourite`, and `visits`, and the region layers in its space.

    Raises:
        ValidationError:
            The populations do not name one number per census region.

    Examples:
        >>> from skcomplex.models.social.organizations import kauai_population
        >>> kauai_population((3, 2, 0)).elements[0].size
        5
    """
    regions = load_island_layers() if layers is None else layers
    counts = load_island_populations() if populations is None else populations
    if len(counts) != len(regions['census']):
        error_msg = f'The populations {counts} need one number for each of the {len(regions["census"])} census regions.'
        raise ValidationError(error_msg)
    rng = np.random.default_rng(random_state)
    census = np.repeat(np.arange(len(counts)), counts)
    size = len(census)
    residents = GroupElement(
        'residents',
        element=ResidentElement,
        size=size,
        values={
            'sentiment': rng.uniform(-0.2, 0.2, size),
            'weights': _get_weights(size, n_topics, weighted_share, rng),
            'home': np.full((size, 2), np.nan),
            'census': census.astype(float),
            'work': rng.integers(len(regions['facilities']), size=size).astype(float),
            'favourite': rng.integers(len(regions['places']), size=size).astype(float),
            'influence': influence,
            'radius': radius,
        },
        location=[regions['census'][region] for region in census.tolist()],
    )
    bounds = shapely.total_bounds([polygon for polygons in regions.values() for polygon in polygons])
    interactions = [
        CommuteInteraction('commute', target='residents'),
        PlaceVisitsInteraction('visits', target='residents'),
        SentimentInteraction('sentiment', target='residents', events=events),
    ]
    space = ContinuousSpace(extent=(float(bounds[2]), float(bounds[3])), regions=regions)
    return System(elements=[residents], interactions=interactions, space=space)
