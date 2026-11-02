"""Patch of grass that regrows logistically every week, `G' = G + g G (1 - G)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RuleLaw
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._pasture_view import PastureView


class PastureElement(Element):
    """Patch of grass that regrows logistically every week, `G' = G + g G (1 - G)`.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        grass:
            The amount of grass, between 0 and 1.

        growth:
            The weekly growth rate `g` of the grass.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.biology.ecology import PastureElement
        >>> GroupElement('pasture', element=PastureElement, size=4, values={'growth': 0.1}).size
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'grass', 'color_range': (0.0, 1.0)}),
        *,
        grass: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        growth: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.grass = grass
        self.growth = growth

    @classmethod
    def _get_view(cls) -> type[PastureView]:
        """Return the view of a patch of pasture, `PastureView`."""
        return PastureView

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

    def regrow(self, agent: PastureView, model: ModelView) -> None:
        """Grow the grass by one week of logistic growth.

        Args:
            agent:
                The patch.

            model:
                The model.
        """
        del model
        agent.grass += agent.growth * agent.grass * (1 - agent.grass)
