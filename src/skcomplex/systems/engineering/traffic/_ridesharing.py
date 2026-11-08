"""Ridesharing: riders who request trips on a city grid and cabs sent to them.

A stable matching sends the cabs.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import LifeCycleInteraction, MapInteraction, MatchingInteraction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.laws.utils import transfer
from ....core.components.spaces import DiscreteTime, LatticeSpace
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.errors import ArgumentError
from ....core.specs import (
    MatchSpec,
    ObservableSpec,
    ParameterSpec,
    PhaseSpec,
    PhasesScheduleSpec,
    RoleSpec,
    StableMatchingSpec,
    StateSpec,
    VisualSpec,
)


class CabView(AgentView):
    """View of a cab of the ridesharing system.

    Its states and parameters are attributes.
    """

    task: float
    target_x: float
    target_y: float
    destination_x: float
    destination_y: float
    rider: float
    seats: float
    money: float
    base_fare: float
    block_fare: float


class RiderView(AgentView):
    """View of a rider of the ridesharing system, with its states as attributes."""

    destination_x: float
    destination_y: float
    waited: float
    seats: float
    money: float


IDLE, PICKING_UP, DELIVERING = 0.0, 1.0, 2.0


class CabElement(Element):
    """Cab that drives a rider to its destination for a fare.

    It drives one block a step to its rider, collects the fare, and drives the rider to its destination. The fare is a
    base fare plus a fare per block of the trip, paid as a transfer from rider to cab.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        task:
            0 when idle, 1 on the way to a rider, and 2 on the way to a destination.

        target_x:
            The column of the cell the cab drives to.

        target_y:
            The row of the cell the cab drives to.

        destination_x:
            The column of the destination of the rider.

        destination_y:
            The row of the destination of the rider.

        rider:
            The id of the rider the cab is sent to.

        seats:
            The free seats of the cab, one when idle and zero on a trip.

        money:
            The fares the cab collected.

        base_fare:
            The fare of every trip.

        block_fare:
            The fare per block of a trip.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.engineering.traffic import CabElement
        >>> GroupElement('cabs', element=CabElement, size=3).size
        3
    """

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'task', 'color_range': (0.0, 2.0)}),
        *,
        task: Annotated[float, StateSpec(domain=NN)] = IDLE,
        target_x: Annotated[float, StateSpec(domain=NN)] = 0.0,
        target_y: Annotated[float, StateSpec(domain=NN)] = 0.0,
        destination_x: Annotated[float, StateSpec(domain=NN)] = 0.0,
        destination_y: Annotated[float, StateSpec(domain=NN)] = 0.0,
        rider: Annotated[float, StateSpec()] = -1.0,
        seats: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        money: Annotated[float, StateSpec()] = 0.0,
        base_fare: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        block_fare: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.task = task
        self.target_x = target_x
        self.target_y = target_y
        self.destination_x = destination_x
        self.destination_y = destination_y
        self.rider = rider
        self.seats = seats
        self.money = money
        self.base_fare = base_fare
        self.block_fare = block_fare

    @classmethod
    def _get_view(cls) -> type[CabView]:
        """Return the view of the cabs.

        It types their states and parameters.
        """
        return CabView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the driving rule of the phase of driving.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.drive, phase='drive')]

    def _pick_up(self, agent: CabView, system: SystemView) -> None:
        """Collect the fare of the rider, take it on board, and head for its destination."""
        rider = next(item for item in system.agents('riders', RiderView) if item.id == int(agent.rider))
        destination = (int(agent.destination_x), int(agent.destination_y))
        blocks = CabElement._blocks(CabElement._cell(agent.location), destination)
        fare = system.parameter('base_fare') + system.parameter('block_fare') * blocks
        transfer(rider, agent, fare, 'money')
        rider.remove()
        new_target_x_value, new_target_y_value = destination
        agent.target_x = new_target_x_value
        agent.target_y = new_target_y_value
        agent.task = DELIVERING

    def drive(self, agent: CabView, system: SystemView) -> None:
        """Drive one block toward the target, and pick up or drop off the rider there.

        Args:
            agent:
                The cab.

            system:
                The system.
        """
        if agent.task == IDLE:
            return
        x, y = CabElement._cell(agent.location)
        target = (int(agent.target_x), int(agent.target_y))
        if (x, y) != target:
            step = (int(np.sign(target[0] - x)), 0) if x != target[0] else (0, int(np.sign(target[1] - y)))
            agent.move_to((x + step[0], y + step[1]))
        if CabElement._cell(agent.location) != target:
            return
        if agent.task == PICKING_UP:
            self._pick_up(agent, system)
        else:
            agent.task = IDLE
            agent.seats = 1.0

    @staticmethod
    def _cell(location: object) -> tuple[int, int]:
        """Return the integer coordinates of a cell."""
        x, y = np.atleast_1d(np.asarray(location, dtype=float))[:2]
        return int(x), int(y)

    @staticmethod
    def _blocks(first: tuple[int, int], second: tuple[int, int]) -> int:
        """Return the number of blocks between two cells of the grid.

        It is the Manhattan distance.
        """
        return abs(first[0] - second[0]) + abs(first[1] - second[1])


class RiderElement(Element):
    """Rider who waits at a cell for a cab to a destination.

    It pays the fare when the cab picks it up.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        destination_x:
            The column of the destination.

        destination_y:
            The row of the destination.

        waited:
            The number of steps the rider has waited.

        seats:
            The seats the rider still requests, one until a cab is on its way and zero after.

        money:
            The money of the rider.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.engineering.traffic import RiderElement
        >>> GroupElement('riders', element=RiderElement, size=2, values={'money': 50.0}).size
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'waited'}),
        *,
        destination_x: Annotated[float, StateSpec(domain=NN)] = 0.0,
        destination_y: Annotated[float, StateSpec(domain=NN)] = 0.0,
        waited: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        seats: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        money: Annotated[float, StateSpec()] = 100.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.destination_x = destination_x
        self.destination_y = destination_y
        self.waited = waited
        self.seats = seats
        self.money = money

    @classmethod
    def _get_view(cls) -> type[RiderView]:
        """Return the view of the riders.

        It types their states.
        """
        return RiderView


class ArrivalInteraction(LifeCycleInteraction):
    """Arrival of new riders at random cells with random destinations.

    It is a life cycle of arrivals only. Every step a Poisson number of riders, with mean `demand`, appear at random
    cells with random destinations.

    Args:
        label:
            The label of the arrivals.

        target:
            The path of the group of riders.

        demand:
            The mean number of new riders in a step.

        width:
            The number of columns of the grid.

        height:
            The number of rows of the grid.

        phase:
            The phase the riders arrive in.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import ArrivalInteraction
        >>> ArrivalInteraction('arrival', target='riders').reports
        ('births', 'deaths', 'arrivals')
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        demand: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        width: int = 30,
        height: int = 30,
        phase: str | None = 'arrival',
    ) -> None:
        super().__init__(
            label,
            target=target,
            placement=self.place,
            values=self.destine,
            arrival=self.expect,
            phase=phase,
        )
        self.demand = demand
        self.width = width
        self.height = height

    def expect(self, system: SystemView) -> float:
        """Return the mean number of new riders in a step.

        Args:
            system:
                The system view of the arrivals.

        Returns:
            demand:
                The demand.
        """
        return system.parameter('demand')

    def place(self, agent: RiderView | None, system: SystemView) -> tuple[int, int]:
        """Return a random cell of the grid for a new rider.

        Args:
            agent:
                The parent, `None` for an arrival.

            system:
                The system view of the arrivals.

        Returns:
            cell:
                The coordinates of the cell.
        """
        x, y = system.rng.integers((self.width, self.height))
        return int(x), int(y)

    def destine(self, agent: RiderView | None, system: SystemView) -> dict[str, object]:
        """Return a random destination on the grid for a new rider.

        Args:
            agent:
                The parent, `None` for an arrival.

            system:
                The system view of the arrivals.

        Returns:
            values:
                The coordinates of the destination by state name.
        """
        to_x, to_y = system.rng.integers((self.width, self.height))
        return {'destination_x': float(to_x), 'destination_y': float(to_y)}


def _busy(agent: CabView) -> float:
    """Return one for a cab on a trip and zero for an idle cab."""
    return float(agent.task != IDLE)


class RidesharingSystem(BaseSystem):
    """Riders who request trips on a city grid and cabs sent to them.

    A stable matching sends the cabs. Every step new riders appear at random cells with random destinations, and the
    dispatch matches the waiting riders with the idle cabs by a stable matching in which riders prefer nearer cabs and
    cabs prefer riders who waited longer. Every cab drives one block a step to its rider, collects the fare, and drives
    the rider to its destination. While the demand stays below the trips the fleet can serve, riders wait about as long
    as a cab takes to reach them. As the demand approaches that capacity, every cab is busy, the queue of waiting riders
    grows, and the waiting time rises steeply, as in a queue near saturation.

    The system has the groups `riders` and `cabs`, the arrivals `arrival`, and the matching `dispatch`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import RidesharingSystem
        >>> [group.size for group in RidesharingSystem(5).prepare().system_.elements]
        [1, 5]
    """

    field = 'engineering/traffic'
    topics = ('ridesharing', 'agents', 'matching', 'queues', 'grid')
    references = ('Carmichael and Hadzikadic 2019 ch. 7',)
    until = 200
    observables = (
        ObservableSpec('queue', group='riders', variable='waited', reduce='count'),
        ObservableSpec('mean_wait', group='riders', variable='waited', reduce='mean'),
        ObservableSpec('utilisation', _busy, group='cabs', reduce='mean'),
    )

    def __init__(
        self,
        n_cabs: int = 30,
        *,
        demand: float = 1.0,
        width: int = 30,
        height: int = 30,
        base_fare: float = 2.0,
        block_fare: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_cabs = n_cabs
        self.demand = demand
        self.width = width
        self.height = height
        self.base_fare = base_fare
        self.block_fare = block_fare
        self.random_state = random_state

    @staticmethod
    def _nearness(rider: object, cab: object) -> float:
        """Return the preference of a rider for a cab, higher for a cab fewer blocks away."""
        if not isinstance(rider, RiderView) or not isinstance(cab, CabView):
            error_msg = f'A rider ranks cabs, but got {rider!r} and {cab!r}.'
            raise ArgumentError(error_msg)
        return -float(CabElement._blocks(CabElement._cell(rider.location), CabElement._cell(cab.location)))

    @staticmethod
    def _patience(cab: object, rider: object) -> float:
        """Return the preference of a cab for a rider.

        It is higher for a rider who waited longer.
        """
        if not isinstance(rider, RiderView):
            error_msg = f'A cab ranks riders, but got {rider!r}.'
            raise ArgumentError(error_msg)
        return rider.waited

    @staticmethod
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

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        cells = rng.integers((self.width, self.height), size=(self.n_cabs + 1, 2))
        riders = GroupElement(
            'riders',
            element=RiderElement,
            size=1,
            values={
                'destination_x': float(rng.integers(self.width)),
                'destination_y': float(rng.integers(self.height)),
            },
            variable=True,
            location=[(int(cells[0, 0]), int(cells[0, 1]))],
        )
        cabs = GroupElement(
            'cabs',
            element=CabElement,
            size=self.n_cabs,
            values={'base_fare': self.base_fare, 'block_fare': self.block_fare},
            location=[(int(x), int(y)) for x, y in cells[1:]],
        )
        return System(
            self.label,
            elements=[riders, cabs],
            interactions=[
                ArrivalInteraction(
                    'arrival',
                    target='riders',
                    demand=self.demand,
                    width=self.width,
                    height=self.height,
                ),
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
                    rule=StableMatchingSpec(requester_preference=self._nearness, provider_preference=self._patience),
                    quantity='seats',
                    phase='dispatch',
                    on_match=self._dispatch,
                ),
            ],
            space=LatticeSpace((self.width, self.height), boundary='fixed', neighborhood='von_neumann'),
            time=DiscreteTime(
                schedule=PhasesScheduleSpec([PhaseSpec('arrival'), PhaseSpec('dispatch'), PhaseSpec('drive')]),
            ),
        )
