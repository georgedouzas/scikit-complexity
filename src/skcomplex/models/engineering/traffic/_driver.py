"""Driver on a ring road who accelerates as much as allowed and stops when the speed would cause a collision."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RuleLaw
from ....engine.solvers import AgentView, ModelView
from ....errors import SimulationError
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._driver_view import DriverView


class DriverElement(Element):
    """Driver on a ring road who accelerates as much as allowed and stops when the speed would cause a collision.

    The speed changes by the acceleration, bounded by `min_acc` and `max_acc`, is multiplied by a random factor between
    `1 - eps` and `1 + eps`, and stays between 0 and the speed limit. A speed larger than the distance to the car ahead
    becomes 0. The car then moves along the ring by its speed, and the ring is as long as the width of the space.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        speed:
            The speed.

        rank:
            The order of the car along the ring.

        acceleration:
            The acceleration every driver chooses.

        min_acc:
            The largest braking, as a negative acceleration.

        max_acc:
            The largest acceleration.

        speed_limit:
            The largest speed.

        eps:
            The relative amplitude of the random error of speed.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.engineering.traffic import DriverElement
        >>> GroupElement('drivers', element=DriverElement, size=2, location=[(0.0, 0.5), (10.0, 0.5)]).size
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'speed'}),
        *,
        speed: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        rank: Annotated[float, StateSpec(domain=NN)] = 0.0,
        acceleration: Annotated[float, ParameterSpec()] = 1.0,
        min_acc: Annotated[float, ParameterSpec()] = -10.0,
        max_acc: Annotated[float, ParameterSpec()] = 1.0,
        speed_limit: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 40.0,
        eps: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
    ) -> None:
        super().__init__(label, visual=visual)
        self.speed = speed
        self.rank = rank
        self.acceleration = acceleration
        self.min_acc = min_acc
        self.max_acc = max_acc
        self.speed_limit = speed_limit
        self.eps = eps

    @classmethod
    def _get_view(cls) -> type[DriverView]:
        """Return the view of the drivers, which types their states and parameters."""
        return DriverView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the driving rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.drive)]

    def drive(self, agent: DriverView, model: ModelView) -> None:
        """Choose the speed from the distance to the car ahead and move along the ring.

        Args:
            agent:
                The driver.

            model:
                The model.
        """
        cars = model.agents(agent.group, DriverView)
        if model.dimensions is None:
            error_msg = 'Drivers move along a ring road, a continuous space with an extent.'
            raise SimulationError(error_msg)
        length = model.dimensions[0]
        ahead = DriverElement._car_ahead(agent, model, len(cars))
        x, y = DriverElement._position(agent)
        distance = (DriverElement._position(ahead)[0] - x) % length if ahead is not None else length
        acceleration = min(max(agent.acceleration, agent.min_acc), agent.max_acc)
        speed = (agent.speed + acceleration) * model.rng.uniform(1 - agent.eps, 1 + agent.eps)
        speed = min(max(speed, 0.0), agent.speed_limit)
        agent.speed = 0.0 if speed > distance else speed
        agent.move_to(((x + agent.speed) % length, y))

    @staticmethod
    def _position(agent: AgentView) -> tuple[float, float]:
        """Return the position of an agent in a continuous space."""
        x, y = np.asarray(agent.location, dtype=float)
        return float(x), float(y)

    @staticmethod
    def _car_ahead(agent: DriverView, model: ModelView, n_cars: int) -> DriverView | None:
        """Return the car whose rank follows the agent's rank on a road of `n_cars` cars, or `None` for the last car."""
        rank = (int(agent.rank) + 1) % n_cars if n_cars else 0
        return next(
            (car for car in model.agents(agent.group, DriverView) if int(car.rank) == rank and car != agent),
            None,
        )
