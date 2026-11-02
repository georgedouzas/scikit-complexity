"""PartElement of a technology, described by its unit cost."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw
from ....specs import StateSpec, VisualSpec
from ._part_view import PartView


class PartElement(Element):
    """PartElement of a technology, described by its unit cost.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        cost:
            The initial unit cost.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.innovation import PartElement
        >>> GroupElement('components', element=PartElement, size=5, values={'cost': 0.5}).size
        5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        cost: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.cost = cost

    @classmethod
    def _get_view(cls) -> type[PartView]:
        """Return the view of the agents of the element."""
        return PartView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since redesigns change several components at once.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []
