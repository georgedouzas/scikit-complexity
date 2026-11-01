"""Run-and-tumble agent in a periodic square, which runs straight and tumbles to a random direction."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RuleLaw
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._tumbler_view import TumblerView


class TumblerElement(Element):
    """Run-and-tumble agent in a periodic square, which runs straight and tumbles to a random direction.

    Every step the agent moves `speed` along its direction, then tumbles with probability `1 - exp(-tumble_rate)`
    to a direction drawn uniformly, as bacteria such as E. coli do.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        theta:
            The initial direction.

        speed:
            The distance run per step.

        tumble_rate:
            The rate of tumbles per step.

        width:
            The side of the periodic square.

    Examples:
        >>> from skcomplex.models.physics.active_matter import TumblerElement
        >>> TumblerElement('bacterium').speed
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'theta'}),
        *,
        theta: Annotated[float, StateSpec()] = 0.0,
        speed: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        tumble_rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        width: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.speed = speed
        self.tumble_rate = tumble_rate
        self.width = width

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the run-and-tumble rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.run_and_tumble)]

    @classmethod
    def _get_view(cls) -> type:
        """Return the class of the views of the element's agents."""
        return TumblerView

    def run_and_tumble(self, agent: TumblerView, model: ModelView) -> None:
        """Run one step along the direction, then tumble at random.

        Args:
            agent:
                The agent.

            model:
                The model.
        """
        position = np.asarray(agent.location, dtype=float)
        heading = np.array([math.cos(agent.theta), math.sin(agent.theta)])
        agent.move_to(np.mod(position + agent.speed * heading, agent.width))
        if model.rng.random() < -math.expm1(-agent.tumble_rate):
            agent.theta = float(model.rng.uniform(0.0, 2 * math.pi))
