"""Linear polymer of an assembly pool, with its length and the step it was made."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec
from ._polymer_view import PolymerView


class PolymerElement(Element):
    """Linear polymer of an assembly pool, with its length and the step it was made.

    Args:
        label:
            The label of the polymer or of the group of polymers.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        length:
            The number of monomers.

        born:
            The step at which the polymer was made.

    Examples:
        >>> from skcomplex.systems.biology.evolution import PolymerElement
        >>> PolymerElement('polymer', length=3.0).length
        3.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        length: Annotated[float, StateSpec(domain=NN)] = 1.0,
        born: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.length = length
        self.born = born

    @classmethod
    def _get_view(cls) -> type[PolymerView]:
        """Return the view of a polymer, `PolymerView`."""
        return PolymerView
