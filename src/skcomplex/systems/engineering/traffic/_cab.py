"""Cab that drives one block a step to its rider, collects the fare, and drives the rider to its destination."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.laws.utils import transfer
from ....core.engine.solvers import SystemView
from ....core.specs import (
    ParameterSpec,
    StateSpec,
    VisualSpec,
)
from ._cab_view import CabView
from ._rider_view import RiderView

IDLE, PICKING_UP, DELIVERING = 0.0, 1.0, 2.0


class CabElement(Element):
    """Cab that drives one block a step to its rider, collects the fare, and drives the rider to its destination.

    The fare is a base fare plus a fare per block of the trip, paid as a transfer from rider to cab.

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
        """Return the view of the cabs, which types their states and parameters."""
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
        """Return the number of blocks between two cells of the grid, the Manhattan distance."""
        return abs(first[0] - second[0]) + abs(first[1] - second[1])
