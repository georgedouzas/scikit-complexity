"""Kauai population system: residents placed by census regions, whose sentiment follows events and the people around."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence

import numpy as np
import shapely
from shapely import Polygon

from ....core.components.elements import GroupElement
from ....core.components.spaces import ContinuousSpace
from ....core.components.system import System
from ....core.errors import ValidationError
from ....core.specs import ObservableSpec
from ..._catalog import register_system
from ._commute import CommuteInteraction
from ._island import load_island_layers, load_island_populations
from ._place_visits import PlaceVisitsInteraction
from ._resident import ResidentElement
from ._sentiment import SentimentInteraction

MIN_WEIGHT = 0.5
OCCUPATION_SHARES = (
    0.48,
    2.80,
    0.54,
    2.65,
    1.22,
    14.97,
    4.97,
    24.25,
    3.63,
    1.66,
    1.41,
    14.26,
    7.69,
    3.31,
    2.85,
    8.09,
    5.20,
)
RACES = 6
GENDERS = 2
OLDEST_AGE = 90.0


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


@register_system(
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
    events: Sequence[tuple[int, int, float]] = ((1, 0, 0.8),),
    convergence: float = 1.0,
    event_rate: float = 0.05,
    cell_size: float = 1000.0,
    partners: int = 15,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return one agent per resident of the census regions of a map, moving between home, work, and favourite places.

    Every resident starts at a uniform point of its census region, its home, and has a facility of work and a favourite
    place drawn uniformly. It commutes between the three in a cycle of three steps. Its sentiment follows the rules of
    Zanbaka et al. in Carmichael et al. (2019 ch. 3 §4.3): the response to events (Eq. 5) and the interaction with a
    partner it chooses from its grid cell (Eqs. 1, 2, and 4), through `SentimentInteraction`. The default map is a
    synthetic island, and the occupations follow Table 1 of the chapter. This system leaves out the base security
    interactions (Eq. 7), physical security information, adverse agents, and the many favourite places and the weekly
    cycle of the chapter. The chapter gives the 15 partners, the grid of 1 km, and the event rate 0.05 of its example,
    and it gives no value of `CR`, the initial sentiments, the topic weights, or the demographics, which are choices of
    this system.

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
            The entries of the events as `(time, topic, weight)`, with the weight in `[-1, 1]`. Entries of one time
            form one event.

        convergence:
            The convergence parameter `CR` of every resident, in `mu = 1 - |a| CR`.

        event_rate:
            The convergence rate `mu` of the response to an event.

        cell_size:
            The side of the grid cells within which residents meet, in the units of the coordinates.

        partners:
            The largest number of residents drawn from the grid cell as possible partners, where 0 turns the interaction
            off.

        random_state:
            The seed or generator of the initial sentiments, the weights, the demographics, the work, and the favourite
            places.

    Returns:
        system:
            The system, with the group `residents`, its states `sentiment`, `weights`, `home`, `census`, `work`,
            `favourite`, `visits`, `race`, `gender`, `occupation`, and `age`, and the region layers in its space.

    Raises:
        ValidationError:
            The populations do not name one number per census region.

    Examples:
        >>> from skcomplex.systems.social.organizations import kauai_population
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
    shares = np.array(OCCUPATION_SHARES) / sum(OCCUPATION_SHARES)
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
            'race': rng.integers(RACES, size=size).astype(float),
            'gender': rng.integers(GENDERS, size=size).astype(float),
            'occupation': rng.choice(len(shares), size=size, p=shares).astype(float),
            'age': rng.uniform(0.0, OLDEST_AGE, size),
            'convergence': convergence,
            'event_rate': event_rate,
            'cell_size': cell_size,
            'partners': float(partners),
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
