"""Kauai population system: residents placed by census regions.

Their sentiment follows events and the people around.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import json
import math
from collections.abc import Mapping, Sequence
from importlib.resources import files
from typing import Annotated, cast

import numpy as np
import numpy.typing as npt
import shapely
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet
from shapely import Polygon
from shapely.geometry import shape

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MapInteraction
from ....core.components.spaces import ContinuousSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import AgentView
from ....core.errors import ValidationError
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array
from ...general.agents import MovementInteraction


class ResidentView(AgentView):
    """View of a resident of the Kauai population system.

    Its states and parameters are attributes.
    """

    sentiment: float
    weights: Array
    home: Array
    census: float
    work: float
    favourite: float
    visits: float
    race: float
    gender: float
    occupation: float
    age: float
    convergence: float
    event_rate: float
    cell_size: float
    partners: float


class ResidentElement(Element):
    """Resident of the Kauai population system.

    The resident lives, works, and spends time at points of a map. A resident starts at its home, a point of its census
    region. It holds a sentiment between -1 and 1, one weight per topic of the events that move the sentiment, and the
    demographic scalars of Carmichael et al. (2019 ch. 3 §4.2).

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        sentiment:
            The initial sentiment.

        weights:
            The weight of every topic.

        home:
            The point of the home, which the first commute sets to the initial location.

        census:
            The index of the census region of the home.

        work:
            The index of the facility of the work.

        favourite:
            The index of the favourite place.

        visits:
            The number of steps the resident stood in its favourite place.

        race:
            The index of the race.

        gender:
            The index of the gender.

        occupation:
            The index of the occupation.

        age:
            The age in years.

        convergence:
            The immutable convergence parameter `CR` of the interactions between agents.

        event_rate:
            The convergence rate `mu` of the response to an event.

        cell_size:
            The side of the grid cells of the space within which residents meet.

        partners:
            The largest number of residents drawn from the grid cell as possible partners.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.organizations import ResidentElement
        >>> GroupElement('residents', element=ResidentElement, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'sentiment', 'color_range': (-1.0, 1.0)}),
        *,
        sentiment: Annotated[float, StateSpec(domain=RealSet.closed(-1, 1))] = 0.0,
        weights: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
        home: Annotated[npt.ArrayLike, StateSpec(vector=True)] = (math.nan, math.nan),
        census: Annotated[float, StateSpec()] = 0.0,
        work: Annotated[float, StateSpec()] = 0.0,
        favourite: Annotated[float, StateSpec()] = 0.0,
        visits: Annotated[float, StateSpec()] = 0.0,
        race: Annotated[float, StateSpec()] = 0.0,
        gender: Annotated[float, StateSpec()] = 0.0,
        occupation: Annotated[float, StateSpec()] = 0.0,
        age: Annotated[float, StateSpec()] = 0.0,
        convergence: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
        event_rate: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.05,
        cell_size: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1000.0,
        partners: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 15.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.sentiment = sentiment
        self.weights = weights
        self.home = home
        self.census = census
        self.work = work
        self.favourite = favourite
        self.visits = visits
        self.race = race
        self.gender = gender
        self.occupation = occupation
        self.age = age
        self.convergence = convergence
        self.event_rate = event_rate
        self.cell_size = cell_size
        self.partners = partners

    @classmethod
    def _get_view(cls) -> type[ResidentView]:
        """Return the view of a resident, `ResidentView`."""
        return ResidentView


AT_WORK = 1
AT_PLACE = 2
STOPS = 3


class CommuteInteraction(MovementInteraction):
    """Commute of every resident in a cycle of three steps.

    The steps go to work, to its favourite place, and back home. The work and the favourite place are regions of the
    layers `facilities` and `places`, so a resident stands at a new uniform point of the region at every visit, and at
    the same point of its home.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of residents.

    Examples:
        >>> from skcomplex.systems.social.organizations import CommuteInteraction
        >>> CommuteInteraction('commute', target='residents').schedule
        'member'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=target, step=self._commute, schedule='member')

    @staticmethod
    def _commute(agent: AgentViewProtocol, system: SystemViewProtocol) -> object:
        """Return the destination of a resident at the current step.

        It is its work, its favourite place, or its home.
        """
        resident = cast(ResidentView, agent)
        if np.isnan(resident.home).any():
            resident.home = np.asarray(resident.location, dtype=float)
        stop = int(system.time) % STOPS
        if stop == AT_WORK:
            return system.region('facilities', int(resident.work))
        if stop == AT_PLACE:
            return system.region('places', int(resident.favourite))
        return resident.home


class PlaceVisitsInteraction(MapInteraction):
    """Count of the steps every resident stands in its favourite place.

    The place is read from the regions of the layer `places`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of residents.

    Examples:
        >>> from skcomplex.systems.social.organizations import PlaceVisitsInteraction
        >>> PlaceVisitsInteraction('visits', target='residents').mode
        'increment'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(
            label,
            target=target,
            state='visits',
            function=self._count,
            mode='increment',
            schedule='member',
            form='agent',
        )

    @staticmethod
    def _count(agent: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return 1 for a resident that stands in its favourite place and 0 for any other."""
        resident = cast(ResidentView, agent)
        return float(int(resident.favourite) in resident.regions_here('places'))


AGE_RANGE = 100.0
SENTIMENT_RANGE = 2.0
DEMOGRAPHIC_TERMS = 5


class SentimentInteraction(MapInteraction):
    """Sentiment of every resident.

    It responds to events and to the partner the resident chooses at each step. The rules are those of Zanbaka et al. in
    Carmichael et al. (2019 ch. 3 §4.3). An event is one or more entries `(time, topic, weight)`, and it changes the
    sentiment `S` to `S + sum(mu A_T (E_T - S))` over its topics, with `mu` the parameter `event_rate` of the resident,
    `A_T` its weight of the topic, and `E_T` the weight of the event (Eq. 5). The partner is the most likely of at most
    `partners` residents drawn from the grid cell of the resident, by `P = DSS / 2 + STI / 2`, with `DSS` the mean of
    five similarities and `STI` the share of topics both weigh (Eqs. 1 and 2). The two sentiments then change at once to
    `a1 + mu1 (a2 - a1)` and `a2 + mu2 (a1 - a2)`, with `mu = 1 - |a| CR` of each and `CR` its parameter `convergence`
    (Eq. 4). The chapter prints no definition of the terms of `DSS`, so here race, gender, and occupation count 1 when
    equal and 0 otherwise, and age and sentiment count one minus their distance over their range. A sentiment stays
    between -1 and 1.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of residents.

        events:
            The entries of the events as `(time, topic, weight)`.

    Examples:
        >>> from skcomplex.systems.social.organizations import SentimentInteraction
        >>> SentimentInteraction('sentiment', target='residents', events=[(1, 0, 0.5)]).events
        [(1, 0, 0.5)]
    """

    draws = True

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        events: Sequence[tuple[int, int, float]] = (),
    ) -> None:
        super().__init__(
            label,
            target=target,
            state='sentiment',
            function=self._update,
            schedule='member',
            form='agent',
        )
        self.events = events

    def _respond_to_events(self, resident: ResidentView, time: int) -> float:
        """Return the sentiment of a resident after the events of a step, by Eq.

        5.
        """
        sentiment = resident.sentiment
        changes = [
            resident.event_rate * resident.weights[topic] * (weight - sentiment)
            for when, topic, weight in self.events
            if when == time
        ]
        return float(np.clip(sentiment + sum(changes), -1.0, 1.0))

    @staticmethod
    def _read_probability(first: ResidentView, second: ResidentView) -> float:
        """Return the probability of interaction of two residents, by Eqs.

        1 and 2.
        """
        similarities = (
            float(first.race == second.race),
            float(first.gender == second.gender),
            float(first.occupation == second.occupation),
            1 - abs(first.age - second.age) / AGE_RANGE,
            1 - abs(first.sentiment - second.sentiment) / SENTIMENT_RANGE,
        )
        shared = np.mean((np.asarray(first.weights) > 0) & (np.asarray(second.weights) > 0))
        return float(sum(similarities) / DEMOGRAPHIC_TERMS / 2 + shared / 2)

    @staticmethod
    def _read_grid_cell(resident: ResidentView) -> tuple[int, ...]:
        """Return the grid cell of the location of a resident."""
        return tuple(np.floor(np.asarray(resident.location, dtype=float) / resident.cell_size).astype(int).tolist())

    def _choose_partner(self, resident: ResidentView, system: SystemViewProtocol) -> ResidentView | None:
        """Return the most likely of the residents drawn from the grid cell of a resident.

        It is `None` without any.
        """
        diagonal = resident.cell_size * np.sqrt(2.0)
        cell = self._read_grid_cell(resident)
        around = [cast(ResidentView, other) for other in resident.neighbors(diagonal)]
        here = [other for other in around if self._read_grid_cell(other) == cell]
        count = min(int(resident.partners), len(here))
        if count == 0:
            return None
        drawn = [here[int(index)] for index in system.rng.choice(len(here), size=count, replace=False)]
        return max(drawn, key=lambda other: self._read_probability(resident, other))

    def _update(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the sentiment of a resident after the step.

        It also sets the sentiment of its partner.
        """
        resident = cast(ResidentView, agent)
        own = self._respond_to_events(resident, int(system.time))
        partner = self._choose_partner(resident, system)
        if partner is None:
            return own
        other = partner.sentiment
        partner.sentiment = other + (1 - abs(other) * partner.convergence) * (own - other)
        return own + (1 - abs(own) * resident.convergence) * (other - own)


PACKAGE = 'skcomplex.systems.social.organizations'
ISLAND_LAYERS = ('boundary', 'census', 'places', 'facilities')


def _read_layer(layer: str) -> str:
    """Return the GeoJSON text of a layer of the synthetic island."""
    return (files(PACKAGE) / f'island_{layer}.geojson').read_text()


def load_island_layers() -> dict[str, list[Polygon]]:
    """Return the polygons of every layer of the synthetic island.

    They are in a metric coordinate system.

    Returns:
        layers:
            The polygons of the layers `boundary`, `census`, `places`, and `facilities`, by the name of the layer.

    Examples:
        >>> from skcomplex.systems.social.organizations import load_island_layers
        >>> {layer: len(polygons) for layer, polygons in load_island_layers().items()}
        {'boundary': 1, 'census': 3, 'places': 3, 'facilities': 2}
    """
    return {
        layer: [shape(item['geometry']) for item in json.loads(_read_layer(layer))['features']]
        for layer in ISLAND_LAYERS
    }


def load_island_populations() -> list[int]:
    """Return the number of residents of every census region of the synthetic island.

    The regions are those of the census layer.

    Returns:
        populations:
            The population of every census region, in the order of the regions of the layer.

    Examples:
        >>> from skcomplex.systems.social.organizations import load_island_populations
        >>> load_island_populations()
        [100, 50, 0]
    """
    return [int(item['properties']['population']) for item in json.loads(_read_layer('census'))['features']]


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


class KauaiPopulationSystem(BaseSystem):
    """One agent per resident of the census regions of a map.

    The agents move between home, work, and favourite places. Every resident starts at a uniform point of its census
    region, its home, and has a facility of work and a favourite place drawn uniformly. It commutes between the three in
    a cycle of three steps. Its sentiment follows the rules of Zanbaka et al. in Carmichael et al. (2019 ch. 3 §4.3):
    the response to events (Eq. 5) and the interaction with a partner it chooses from its grid cell (Eqs. 1, 2, and 4),
    through `SentimentInteraction`. The default map is a synthetic island, and the occupations follow Table 1 of the
    chapter. This system leaves out the base security interactions (Eq. 7), physical security information, adverse
    agents, and the many favourite places and the weekly cycle of the chapter. The chapter gives the 15 partners, the
    grid of 1 km, and the event rate 0.05 of its example, and it gives no value of `CR`, the initial sentiments, the
    topic weights, or the demographics, which are choices of this system.

    The system has the group `residents`, its states `sentiment`, `weights`, `home`, `census`, `work`, `favourite`,
    `visits`, `race`, `gender`, `occupation`, and `age`, and the region layers in its space.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The populations do not name one number per census region.

    Examples:
        >>> from skcomplex.systems.social.organizations import KauaiPopulationSystem
        >>> KauaiPopulationSystem((3, 2, 0)).prepare().system_.elements[0].size
        5
    """

    field = 'social/organizations'
    topics = ('agents', 'continuous space', 'region layers', 'census', 'sentiment')
    references = ('Carmichael et al. 2019',)
    until = 30
    observables = (ObservableSpec('mean_sentiment', group='residents', variable='sentiment', reduce='mean'),)

    def __init__(
        self,
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
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.populations = populations
        self.layers = layers
        self.n_topics = n_topics
        self.weighted_share = weighted_share
        self.events = events
        self.convergence = convergence
        self.event_rate = event_rate
        self.cell_size = cell_size
        self.partners = partners
        self.random_state = random_state

    @staticmethod
    def _get_weights(n_residents: int, n_topics: int, share: float, rng: np.random.Generator) -> np.ndarray:
        """Return the weight of every topic per resident.

        The weight is above zero for the given share of residents of every topic.
        """
        weights = np.zeros((n_residents, n_topics))
        n_weighted = round(share * n_residents)
        for topic in range(n_topics):
            weighted = rng.permutation(n_residents)[:n_weighted]
            weights[weighted, topic] = rng.uniform(MIN_WEIGHT, 1.0, n_weighted)
        return weights

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        regions = load_island_layers() if self.layers is None else self.layers
        counts = load_island_populations() if self.populations is None else self.populations
        if len(counts) != len(regions['census']):
            error_msg = (
                f'The populations {counts} need one number for each of the {len(regions["census"])} census regions.'
            )
            raise ValidationError(error_msg)
        rng = np.random.default_rng(self.random_state)
        census = np.repeat(np.arange(len(counts)), counts)
        size = len(census)
        shares = np.array(OCCUPATION_SHARES) / sum(OCCUPATION_SHARES)
        residents = GroupElement(
            'residents',
            element=ResidentElement,
            size=size,
            values={
                'sentiment': rng.uniform(-0.2, 0.2, size),
                'weights': self._get_weights(size, self.n_topics, self.weighted_share, rng),
                'home': np.full((size, 2), np.nan),
                'census': census.astype(float),
                'work': rng.integers(len(regions['facilities']), size=size).astype(float),
                'favourite': rng.integers(len(regions['places']), size=size).astype(float),
                'race': rng.integers(RACES, size=size).astype(float),
                'gender': rng.integers(GENDERS, size=size).astype(float),
                'occupation': rng.choice(len(shares), size=size, p=shares).astype(float),
                'age': rng.uniform(0.0, OLDEST_AGE, size),
                'convergence': self.convergence,
                'event_rate': self.event_rate,
                'cell_size': self.cell_size,
                'partners': float(self.partners),
            },
            location=[regions['census'][region] for region in census.tolist()],
        )
        bounds = shapely.total_bounds([polygon for polygons in regions.values() for polygon in polygons])
        interactions = [
            CommuteInteraction('commute', target='residents'),
            PlaceVisitsInteraction('visits', target='residents'),
            SentimentInteraction('sentiment', target='residents', events=self.events),
        ]
        space = ContinuousSpace(extent=(float(bounds[2]), float(bounds[3])), regions=regions)
        return CompositeSystem(self.label, elements=[residents], interactions=interactions, space=space)
