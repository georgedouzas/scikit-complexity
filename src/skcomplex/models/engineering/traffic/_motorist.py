"""Car of Charbonneau's traffic model, which brakes when it is closer than a safe distance to the car ahead."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RuleLaw
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._motorist_view import MotoristView


class MotoristElement(Element):
    """Car of Charbonneau's traffic model, which brakes when it is closer than a safe distance to the car ahead.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The position along the road.

        v:
            The speed.

        rank:
            The order of the car along the road, 0 for the last car.

        safe_distance:
            The distance to the car ahead below which the driver brakes.

        braking:
            The speed lost when braking.

        acceleration:
            The speed gained when accelerating.

        speed_limit:
            The largest speed.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.engineering.traffic import MotoristElement
        >>> GroupElement('cars', element=MotoristElement, size=3, values={'x': [0.0, 10.0, 20.0]}).size
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'position': 'x', 'color_by': 'v'}),
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        v: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        rank: Annotated[float, StateSpec(domain=NN)] = 0.0,
        safe_distance: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        braking: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
        acceleration: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        speed_limit: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v
        self.rank = rank
        self.safe_distance = safe_distance
        self.braking = braking
        self.acceleration = acceleration
        self.speed_limit = speed_limit

    @classmethod
    def _get_view(cls) -> type[MotoristView]:
        """Return the view of the cars, which types their states and parameters."""
        return MotoristView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the speed adjustment rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.adjust)]

    def adjust(self, agent: MotoristView, model: ModelView) -> None:
        """Brake when the car ahead is closer than the safe distance and accelerate otherwise.

        The lead car, which has no car ahead, keeps its distance to the car behind it instead.

        Args:
            agent:
                The car.

            model:
                The model.
        """
        cars = model.agents(agent.group, MotoristView)
        ahead = next((car for car in cars if int(car.rank) == int(agent.rank) + 1), None)
        if ahead is None:
            behind = next((car for car in cars if int(car.rank) == int(agent.rank) - 1), None)
            distance = agent.x - behind.x if behind is not None else np.inf
        else:
            distance = ahead.x - agent.x
        change = -agent.braking if distance < agent.safe_distance else agent.acceleration
        agent.v = float(np.clip(agent.v + change, 0.0, agent.speed_limit))
