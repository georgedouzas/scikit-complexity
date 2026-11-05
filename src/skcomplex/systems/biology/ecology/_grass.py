"""Grass patch that regrows a fixed number of steps after it is eaten."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._grass_view import GrassView


class GrassElement(Element):
    """Grass patch that regrows a fixed number of steps after it is eaten.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        grown:
            Whether the grass is grown, 1 or 0.

        countdown:
            The steps left until the grass is grown again.

        regrowth:
            The number of steps the grass takes to regrow.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.ecology import GrassElement
        >>> GroupElement('grass', element=GrassElement, size=4, values={'grown': 1.0}).size
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec(
            'agents',
            {'color_by': 'grown', 'color_range': (0.0, 1.0), 'radius': 0.5},
        ),
        *,
        grown: Annotated[float, StateSpec(domain=NN)] = 1.0,
        countdown: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        regrowth: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 30.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.grown = grown
        self.countdown = countdown
        self.regrowth = regrowth

    @classmethod
    def _get_view(cls) -> type[GrassView]:
        """Return the view of a patch of grass, `GrassView`."""
        return GrassView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the regrowth rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.regrow)]

    def regrow(self, agent: GrassView, system: SystemView) -> None:
        """Count down eaten grass and grow it back when the countdown ends.

        Args:
            agent:
                The patch.

            system:
                The system.
        """
        del system
        if agent.grown:
            return
        agent.countdown -= 1
        if agent.countdown <= 0:
            agent.grown = 1.0
            agent.countdown = agent.regrowth
