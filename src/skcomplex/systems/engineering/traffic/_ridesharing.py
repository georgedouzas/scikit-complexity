"""Ridesharing: riders who request trips on a city grid and cabs that a stable matching sends to them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.interactions import MapInteraction, MatchingInteraction
from ....core.components.spaces import DiscreteTime, LatticeSpace
from ....core.components.system import System
from ....core.engine.solvers import SystemView
from ....core.errors import ArgumentError
from ....core.specs import (
    MatchSpec,
    ObservableSpec,
    PhaseSpec,
    PhasesScheduleSpec,
    StableMatchingSpec,
)
from ..._catalog import register_system
from ._arrival import ArrivalInteraction
from ._cab import IDLE, PICKING_UP, CabElement
from ._cab_view import CabView
from ._rider import RiderElement
from ._rider_view import RiderView


def _busy(agent: CabView) -> float:
    """Return one for a cab on a trip and zero for an idle cab."""
    return float(agent.task != IDLE)


def _ridesharing_observables(system: System) -> list[ObservableSpec]:
    """Return the number of waiting riders, their mean waiting time, and the share of busy cabs."""
    del system
    return [
        ObservableSpec('queue', group='riders', variable='waited', reduce='count'),
        ObservableSpec('mean_wait', group='riders', variable='waited', reduce='mean'),
        ObservableSpec('utilisation', _busy, group='cabs', reduce='mean'),
    ]


def _nearness(rider: object, cab: object) -> float:
    """Return the preference of a rider for a cab, higher for a cab fewer blocks away."""
    if not isinstance(rider, RiderView) or not isinstance(cab, CabView):
        error_msg = f'A rider ranks cabs, but got {rider!r} and {cab!r}.'
        raise ArgumentError(error_msg)
    return -float(CabElement._blocks(CabElement._cell(rider.location), CabElement._cell(cab.location)))


def _patience(cab: object, rider: object) -> float:
    """Return the preference of a cab for a rider, higher for a rider who waited longer."""
    if not isinstance(rider, RiderView):
        error_msg = f'A cab ranks riders, but got {rider!r}.'
        raise ArgumentError(error_msg)
    return rider.waited


def _dispatch(ride: MatchSpec, system: SystemView) -> None:
    """Send the cab of a ride to its rider."""
    del system
    rider, cab = ride.requester, ride.provider
    if not isinstance(rider, RiderView) or not isinstance(cab, CabView):
        error_msg = f'A ride matches a rider with a cab, but got {rider!r} and {cab!r}.'
        raise ArgumentError(error_msg)
    new_target_x_value, new_target_y_value = CabElement._cell(rider.location)
    cab.target_x = new_target_x_value
    cab.target_y = new_target_y_value
    new_destination_x, new_destination_y = rider.destination_x, rider.destination_y
    cab.destination_x = new_destination_x
    cab.destination_y = new_destination_y
    cab.rider = float(rider.id)
    cab.task = PICKING_UP


@register_system(
    field='engineering/traffic',
    tags=('ridesharing', 'agents', 'matching', 'queues', 'grid'),
    references=('Carmichael and Hadzikadic 2019 ch. 7',),
    defaults={'until': 200},
    observables=_ridesharing_observables,
)
def ridesharing(
    n_cabs: int = 30,
    *,
    demand: float = 1.0,
    width: int = 30,
    height: int = 30,
    base_fare: float = 2.0,
    block_fare: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return riders who request trips on a city grid and cabs that a stable matching sends to them.

    Every step new riders appear at random cells with random destinations, and the dispatch matches the waiting
    riders with the idle cabs by a stable matching in which riders prefer nearer cabs and cabs prefer riders who waited
    longer. Every cab drives one block a step to its rider, collects the fare, and drives the rider to its destination.
    While the demand stays below the trips the fleet can serve, riders wait about as long as a cab takes to reach them.
    As the demand approaches that capacity, every cab is busy, the queue of waiting riders grows, and the waiting time
    rises steeply, as in a queue near saturation.

    Args:
        n_cabs:
            The number of cabs.

        demand:
            The mean number of new riders in a step.

        width:
            The number of columns of the grid.

        height:
            The number of rows of the grid.

        base_fare:
            The fare of every trip.

        block_fare:
            The fare per block of a trip.

        random_state:
            The seed or generator of the cells of the cabs.

    Returns:
        system:
            The system, with the groups `riders` and `cabs`, the arrivals `arrival`, and the matching `dispatch`.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import ridesharing
        >>> [group.size for group in ridesharing(5).elements]
        [1, 5]
    """
    rng = np.random.default_rng(random_state)
    cells = rng.integers((width, height), size=(n_cabs + 1, 2))
    riders = GroupElement(
        'riders',
        element=RiderElement,
        size=1,
        values={'destination_x': float(rng.integers(width)), 'destination_y': float(rng.integers(height))},
        variable=True,
        location=[(int(cells[0, 0]), int(cells[0, 1]))],
    )
    cabs = GroupElement(
        'cabs',
        element=CabElement,
        size=n_cabs,
        values={'base_fare': base_fare, 'block_fare': block_fare},
        location=[(int(x), int(y)) for x, y in cells[1:]],
    )
    return System(
        elements=[riders, cabs],
        interactions=[
            ArrivalInteraction('arrival', target='riders', demand=demand, width=width, height=height),
            MapInteraction(
                'waiting',
                target='riders',
                state='waited',
                function=lambda agent, system: 1.0,
                mode='increment',
                form='agent',
                phase='drive',
            ),
            MatchingInteraction(
                'dispatch',
                requesters='riders',
                providers='cabs',
                rule=StableMatchingSpec(requester_preference=_nearness, provider_preference=_patience),
                quantity='seats',
                phase='dispatch',
                on_match=_dispatch,
            ),
        ],
        space=LatticeSpace((width, height), boundary='fixed', neighborhood='von_neumann'),
        time=DiscreteTime(
            schedule=PhasesScheduleSpec([PhaseSpec('arrival'), PhaseSpec('dispatch'), PhaseSpec('drive')]),
        ),
    )
