"""Global inhibitor of a LEGION network, whose activity `z` decays at rate `phi` without drive."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class InhibitorElement(Element):
    """Global inhibitor of a LEGION network, whose activity `z` decays at rate `phi` without drive.

    Args:
        label:
            The label of the inhibitor.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        z:
            The initial activity.

        phi:
            The rate of the inhibitor.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import InhibitorElement
        >>> InhibitorElement('inhibitor').phi
        3.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        z: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        phi: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.z = z
        self.phi = phi

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the decay of the inhibitor.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.z, -s.phi * s.z)]
