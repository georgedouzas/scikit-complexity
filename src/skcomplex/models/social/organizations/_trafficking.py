"""Trafficking models: victims moved between brothels and the traces their movements leave."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.elements import GroupElement
from ....core.interactions import MapInteraction
from ....core.spaces import DiscreteTime, LatticeSpace
from ....core.system import System
from ....core.types import ModelViewProtocol
from ....specs import PhaseSpec, PhasesScheduleSpec
from ..._catalog import register_model
from ._daily_movements import DailyMovementsInteraction
from ._premises import PremisesElement
from ._premises_view import PremisesView
from ._traveller import TravellerElement

MONTH = 30


QUARTER = 90


DAILY = 'default'


MONTHLY = 'trafficking'


BILLING = 'billing'


QUARTERLY = 'quarter'


def _heat(place: PremisesView, model: ModelViewProtocol) -> float:
    """Return the growth of the bills of a premises in a day, `1 / (1 + exp(x0 - n))` for the `n` people inside."""
    people = model.occupancy('sources').get(place.location_key, 0)
    return float(1 / (1 + np.exp(place.efficiency - people)))


def _count_day(place: PremisesView, model: ModelViewProtocol) -> float:
    """Return one day."""
    return 1.0


def _record(place: PremisesView, model: ModelViewProtocol) -> float:
    """Return the bills of the quarter that ends."""
    return float(place.bills)


def _reset(place: PremisesView, model: ModelViewProtocol) -> float:
    """Return the bills at the start of a quarter."""
    return 0.0


@register_model(
    field='social/organizations',
    tags=('trafficking', 'agents', 'grid', 'crime'),
    references=('Chesney 2021',),
    defaults={'until': 90},
)
def trafficking(
    width: int = 30,
    height: int = 30,
    *,
    n_cities: int = 3,
    n_houses: int = 40,
    n_brothels: int = 3,
    n_victims: int = 12,
    n_free: int = 30,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the brothel movement model, where victims moved between brothels leave traces in bills and train trips.

    Args:
        width:
            The number of columns of the grid.

        height:
            The number of rows of the grid.

        n_cities:
            The number of cities, vertical bands of equal width.

        n_houses:
            The number of houses.

        n_brothels:
            The number of brothels.

        n_victims:
            The number of victims, spread over the brothels.

        n_free:
            The number of women who travel freely.

        random_state:
            The seed or generator of the places, the women, and the efficiencies.

    Returns:
        system:
            The model, with the groups `premises` and `women`.

    Examples:
        >>> from skcomplex.models.social.organizations import trafficking
        >>> [group.size for group in trafficking(n_houses=5, n_brothels=2, n_victims=4, n_free=3).elements]
        [7, 7]
    """
    rng = np.random.default_rng(random_state)
    n_places = n_houses + n_brothels
    cells = rng.permutation(width * height)[:n_places]
    places = [(int(cell % width), int(cell // width)) for cell in cells]
    brothel = (np.arange(n_places) >= n_houses).astype(float)
    premises = GroupElement(
        'premises',
        element=PremisesElement,
        size=n_places,
        values={'brothel': brothel, 'efficiency': rng.uniform(0, 3, n_places)},
        location=places,
    )
    homes = [places[n_houses + index % n_brothels] for index in range(n_victims)]
    homes += [(int(rng.integers(width)), int(rng.integers(height))) for _ in range(n_free)]
    victim = (np.arange(n_victims + n_free) < n_victims).astype(float)
    women = GroupElement(
        'women',
        element=TravellerElement,
        size=n_victims + n_free,
        values={'victim': victim},
        location=homes,
    )
    day = DailyMovementsInteraction(
        'day',
        women='women',
        premises='premises',
        n_cities=n_cities,
        phase=DAILY,
        victims_phase=MONTHLY,
    )
    bills = [
        MapInteraction(
            'days',
            target='premises',
            state='day',
            function=_count_day,
            mode='increment',
            form='agent',
            phase=BILLING,
        ),
        MapInteraction(
            'bills',
            target='premises',
            state='bills',
            function=_heat,
            sources=['women'],
            mode='increment',
            form='agent',
            phase=BILLING,
        ),
        MapInteraction(
            'record',
            target='premises',
            state='last_bills',
            function=_record,
            form='agent',
            phase=QUARTERLY,
        ),
        MapInteraction('reset', target='premises', state='bills', function=_reset, form='agent', phase=QUARTERLY),
    ]
    phases = [
        PhaseSpec(DAILY, activation='sequential'),
        PhaseSpec(MONTHLY, activation='sequential', period=MONTH),
        PhaseSpec(BILLING, activation='sequential'),
        PhaseSpec(QUARTERLY, activation='sequential', period=QUARTER),
    ]
    return System(
        elements=[premises, women],
        interactions=[day, *bills],
        space=LatticeSpace((width, height), boundary='fixed'),
        time=DiscreteTime(schedule=PhasesScheduleSpec(phases)),
    )
