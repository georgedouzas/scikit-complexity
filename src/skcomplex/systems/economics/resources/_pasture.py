"""Cell of the meadow whose grass grows back logistically, from a seed bank when it is grazed bare."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import SystemView
from ....core.specs import (
    ParameterSpec,
    StateSpec,
    VisualSpec,
)
from ._pasture_view import PastureView


class PastureElement(Element):
    """Cell of the meadow whose grass grows back logistically, from a seed bank when it is grazed bare.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        grass:
            The grass on the cell.

        capacity:
            The most grass the cell holds.

        regrowth:
            The logistic growth rate of the grass.

        seed:
            The share of the capacity that grows back as if it stood on a bare cell.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.resources import PastureElement
        >>> GroupElement('meadow', element=PastureElement, size=4, values={'capacity': 2.0}).size
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'grass', 'color_range': (0.0, 2.0)}),
        *,
        grass: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        capacity: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 2.0,
        regrowth: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        seed: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.grass = grass
        self.capacity = capacity
        self.regrowth = regrowth
        self.seed = seed

    @classmethod
    def _get_view(cls) -> type[PastureView]:
        """Return the view of the agents of the element."""
        return PastureView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the regrowth rule of the phase of growth.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.grow, phase='growth')]

    def grow(self, agent: PastureView, system: SystemView) -> None:
        """Grow the grass logistically up to the capacity.

        Args:
            agent:
                The pasture.

            system:
                The system.
        """
        stand = max(agent.grass, system.parameter('seed') * agent.capacity)
        growth = system.parameter('regrowth') * stand * (1 - stand / agent.capacity)
        agent.grass = min(agent.capacity, agent.grass + growth)
