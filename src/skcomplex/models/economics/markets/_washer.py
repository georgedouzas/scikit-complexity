"""Worker who joins the front car of the wash or of the valet while it has fewer than two workers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import ClassVar

from ....core.elements import Element
from ....core.laws import BaseLaw, RuleLaw
from ....engine.solvers import AgentView, ModelView
from ._car import WASH
from ._car_view import CarView

VALET = 1


CREW_SIZE = 2


class WasherElement(Element):
    """Worker who joins the front car of the wash or of the valet while it has fewer than two workers.

    A worker who finds both areas staffed helps at a random one of them. One second of work removes one
    person-second from the car.

    Args:
        label:
            The label of the agent or group.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.markets import WasherElement
        >>> GroupElement('workers', element=WasherElement, size=4).size
        4
    """

    cars: ClassVar[str] = 'cars'

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the work rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.work)]

    def work(self, agent: AgentView, model: ModelView) -> None:
        """Work one second on the front car of an area.

        Args:
            agent:
                The worker.

            model:
                The model.
        """
        cars = model.agents(self.cars, CarView)
        fronts = [
            car for car in (WasherElement._front(cars, WASH), WasherElement._front(cars, VALET)) if car is not None
        ]
        if not fronts:
            return
        short = [car for car in fronts if car.crew < CREW_SIZE]
        car = short[0] if short else fronts[int(model.rng.integers(len(fronts)))]
        car.crew += 1
        car.work -= 1.0

    @staticmethod
    def _front(cars: list[CarView], stage: int) -> CarView | None:
        """Return the car that came first among the cars at a stage, or `None` if there is none."""
        waiting = [car for car in cars if car.stage == stage]
        return min(waiting, key=lambda car: car.id) if waiting else None
