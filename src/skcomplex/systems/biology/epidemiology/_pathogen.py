"""Pathogen whose strength in every season is drawn anew, uniformly from zero to one."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import uniform
from ....core.specs import StateSpec, VisualSpec


class PathogenElement(Element):
    """Pathogen whose strength in every season is drawn anew, uniformly from zero to one.

    Args:
        label:
            The label of the pathogen.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        eta:
            The strength of the strain of the current season.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import PathogenElement
        >>> PathogenElement('pathogen').eta
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        eta: Annotated[float, StateSpec(domain=RealSet.closed_open(0, 1))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.eta = eta

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the strength of the next season, a uniform draw.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [UpdateLaw(s.eta, uniform('eta'))]
