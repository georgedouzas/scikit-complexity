"""Arrival of new riders at random cells with random destinations, a life cycle of arrivals only."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import LifeCycleInteraction
from ....core.engine.solvers import SystemView
from ....core.specs import (
    ParameterSpec,
    RoleSpec,
)
from ._rider_view import RiderView


class ArrivalInteraction(LifeCycleInteraction):
    """Arrival of new riders at random cells with random destinations, a life cycle of arrivals only.

    Every step a Poisson number of riders, with mean `demand`, appear at random cells with random destinations.

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
