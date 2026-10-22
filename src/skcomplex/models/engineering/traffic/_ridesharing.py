"""Ridesharing: riders who request trips on a city grid and cabs that a dispatcher sends to the nearest of them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ....core.backends import AgentView, ModelView
from ....core.base import Domain, VisualSpec, parameter, role, state
from ....core.components import Element
from ....core.interactions import Interaction, Match, matches
from ....core.laws import Law, Rule, Symbols, transfer
from ....core.runtime import register_model
from ....core.space import DiscreteTime, LatticeSpace, Phase, Phases
from ....core.system import Observable, System

IDLE, PICKING_UP, DELIVERING = 0.0, 1.0, 2.0


def _cell(location: object) -> tuple[int, int]:
    """Return the integer coordinates of a cell."""
    x, y = np.atleast_1d(np.asarray(location, dtype=float))[:2]
    return int(x), int(y)


def _blocks(first: tuple[int, int], second: tuple[int, int]) -> int:
    """Return the number of blocks between two cells of the grid, the Manhattan distance."""
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def _busy(agent: Any) -> float:  # noqa: ANN401
    """Return one for a cab on a trip and zero for an idle cab."""
    return float(agent.task != IDLE)


def _ridesharing_observables(system: System) -> list[Observable]:
    """Return the number of waiting riders, their mean waiting time, and the share of busy cabs."""
    del system
    return [
        Observable('queue', group='riders', variable='waited', reduce='count'),
        Observable('mean_wait', group='riders', variable='waited', reduce='mean'),
        Observable('utilisation', _busy, group='cabs', reduce='mean'),
    ]


@dataclass(frozen=True)
class NearestCab:
    """Pairing of unassigned riders, the longest waiting first, with the nearest idle cab by blocks.

    Examples:
        >>> from skcomplex.models.engineering.traffic import NearestCab
        >>> NearestCab()
        NearestCab()
    """

    def __call__(
        self,
        left: Sequence[Any],
        right: Sequence[Any],
        rng: np.random.Generator,
        capacity: int,
    ) -> list[tuple[Any, Any]]:
        """Return the pairs of riders and cabs.

        Args:
            left:
                The riders.

            right:
                The cabs.

            rng:
                The random generator, which breaks ties between riders that waited as long.

            capacity:
                The largest number of riders of a cab, which a cab on a trip does not take.

        Returns:
            pairs:
                The pairs of riders and the cabs sent to them.
        """
        riders = [rider for rider in left if not rider.assigned]
        order = np.lexsort((rng.random(len(riders)), [-rider.waited for rider in riders])) if riders else []
        free = [cab for cab in right if cab.task == IDLE]
        pairs: list[tuple[Any, Any]] = []
        for index in order:
            if not free:
                break
            rider = riders[int(index)]
            cell = _cell(rider.location)
            nearest = min(range(len(free)), key=lambda item: _blocks(cell, _cell(free[item].location)))
            pairs.append((rider, free.pop(nearest)))
        return pairs


class Rider(Element):
    """Rider who waits at a cell for a cab to a destination, and pays the fare when the cab picks it up.

    Args:
        label:
            The label of the agent or group.

        destination_x:
            The column of the destination.

        destination_y:
            The row of the destination.

        waited:
            The number of steps the rider has waited.

        assigned:
            One once a cab is on its way, zero before.

        money:
            The money of the rider.

    Examples:
        >>> from skcomplex.models.engineering.traffic import Rider
        >>> Rider.group('riders', 2, money=50.0).size
        2
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'waited'})
    destination_x: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    destination_y: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    waited: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    assigned: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    money: float = state(default=100.0, domain=Domain.REAL)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the waiting rule of the phase of driving.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.wait, phase='drive')]

    def wait(self, agent: AgentView, model: ModelView) -> None:
        """Wait one more step.

        Args:
            agent:
                The rider.

            model:
                The model.
        """
        agent.waited += 1


class Cab(Element):
    """Cab that drives one block a step to its rider, collects the fare, and drives the rider to its destination.

    The fare is a base fare plus a fare per block of the trip, paid as a transfer from rider to cab.

    Args:
        label:
            The label of the agent or group.

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

        money:
            The fares the cab collected.

        base_fare:
            The fare of every trip.

        block_fare:
            The fare per block of a trip.

    Examples:
        >>> from skcomplex.models.engineering.traffic import Cab
        >>> Cab.group('cabs', 3).size
        3
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'task', 'color_range': (0.0, 2.0)})
    task: float = state(default=IDLE, domain=Domain.NONNEGATIVE_INTEGER)
    target_x: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    target_y: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    destination_x: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    destination_y: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    rider: float = state(default=-1.0, domain=Domain.REAL)
    money: float = state(default=0.0, domain=Domain.REAL)
    base_fare: float = parameter(default=2.0, domain=Domain.NONNEGATIVE)
    block_fare: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the driving rule of the phase of driving.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.drive, phase='drive')]

    def _pick_up(self, agent: AgentView, model: ModelView) -> None:
        """Collect the fare of the rider, take it on board, and head for its destination."""
        rider = next(item for item in model.agents('riders') if item.id == int(agent.rider))
        destination = (int(agent.destination_x), int(agent.destination_y))
        blocks = _blocks(_cell(agent.location), destination)
        fare = model.parameter('base_fare') + model.parameter('block_fare') * blocks
        transfer(rider, agent, fare, 'money')
        rider.remove()
        agent.target_x, agent.target_y = destination
        agent.task = DELIVERING

    def drive(self, agent: AgentView, model: ModelView) -> None:
        """Drive one block toward the target, and pick up or drop off the rider there.

        Args:
            agent:
                The cab.

            model:
                The model.
        """
        if agent.task == IDLE:
            return
        x, y = _cell(agent.location)
        target = (int(agent.target_x), int(agent.target_y))
        if (x, y) != target:
            step = (int(np.sign(target[0] - x)), 0) if x != target[0] else (0, int(np.sign(target[1] - y)))
            agent.move_to((x + step[0], y + step[1]))
        if _cell(agent.location) != target:
            return
        if agent.task == PICKING_UP:
            self._pick_up(agent, model)
        else:
            agent.task = IDLE


class Dispatch(Interaction):
    """Dispatcher that admits new riders and sends the nearest idle cab to every waiting rider.

    Every step a Poisson number of riders, with mean `demand`, appear at random cells with random destinations. The
    matching law pairs the waiting riders, the longest waiting first, with the nearest idle cabs, and every paired cab
    sets off to its rider.

    Args:
        label:
            The label of the dispatcher.

        riders:
            The path of the group of riders.

        cabs:
            The path of the group of cabs.

        demand:
            The mean number of new riders in a step.

        width:
            The number of columns of the grid.

        height:
            The number of rows of the grid.

    Examples:
        >>> from skcomplex.models.engineering.traffic import Dispatch
        >>> Dispatch('dispatch', riders='riders', cabs='cabs').reports
        ('arrivals',)
    """

    reports: ClassVar[tuple[str, ...]] = ('arrivals',)
    riders: str = role()
    cabs: str = role()
    demand: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    width: int = 30
    height: int = 30

    def laws(self, s: Symbols) -> list[Any]:
        """Return the arrival of riders, the matching of riders with cabs, and the rule that dispatches the cabs.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            Rule(self.admit, phase='arrival'),
            Match('riders', 'cabs', pairing=NearestCab(), label='rides', phase='dispatch'),
            Rule(self.dispatch, phase='dispatch'),
        ]

    def admit(self, model: Any) -> None:  # noqa: ANN401
        """Create the riders of the step at random cells with random destinations.

        Args:
            model:
                The model view of the dispatcher.
        """
        arrivals = int(model.rng.poisson(model.parameter('demand')))
        for _ in range(arrivals):
            x, y, to_x, to_y = model.rng.integers((self.width, self.height, self.width, self.height))
            model.create('riders', location=(int(x), int(y)), destination_x=float(to_x), destination_y=float(to_y))
        model.report('arrivals', arrivals)

    def dispatch(self, model: Any) -> None:  # noqa: ANN401
        """Send every paired cab to its rider.

        Args:
            model:
                The model view of the dispatcher.
        """
        for rider, cab in matches(model, 'rides'):
            cab.target_x, cab.target_y = _cell(rider.location)
            cab.destination_x, cab.destination_y = rider.destination_x, rider.destination_y
            cab.rider = float(rider.id)
            cab.task = PICKING_UP
            rider.assigned = 1.0


@register_model(
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
    """Return riders who request trips on a city grid and cabs that a dispatcher sends to the nearest of them.

    Every step new riders appear at random cells with random destinations, the dispatcher pairs the waiting riders,
    the longest waiting first, with the nearest idle cabs, and every cab drives one block a step to its rider, collects
    the fare, and drives the rider to its destination. While the demand stays below the trips the fleet can serve,
    riders wait about as long as a cab takes to reach them. As the demand approaches that capacity, every cab is busy,
    the queue of waiting riders grows, and the waiting time rises steeply, as in a queue near saturation.

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
            The model, with the groups `riders` and `cabs` and the dispatcher `dispatch`.

    Examples:
        >>> from skcomplex.models.engineering.traffic import ridesharing
        >>> [group.size for group in ridesharing(5).elements]
        [1, 5]
    """
    rng = np.random.default_rng(random_state)
    cells = rng.integers((width, height), size=(n_cabs + 1, 2))
    riders = Rider.group(
        'riders',
        1,
        variable=True,
        location=[(int(cells[0, 0]), int(cells[0, 1]))],
        destination_x=float(rng.integers(width)),
        destination_y=float(rng.integers(height)),
    )
    cabs = Cab.group(
        'cabs',
        n_cabs,
        location=[(int(x), int(y)) for x, y in cells[1:]],
        base_fare=base_fare,
        block_fare=block_fare,
    )
    return System(
        components=[riders, cabs],
        interactions=[Dispatch('dispatch', riders='riders', cabs='cabs', demand=demand, width=width, height=height)],
        space=LatticeSpace((width, height), boundary='fixed', neighborhood='von_neumann'),
        time=DiscreteTime(schedule=Phases([Phase('arrival'), Phase('dispatch'), Phase('drive')])),
    )
