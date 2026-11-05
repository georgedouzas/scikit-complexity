"""Bank of the DebtRank cascade, with its level of distress, its state, and its economic value."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw
from ....core.specs import StateSpec, VisualSpec
from ._interbank_view import InterbankView

UNDISTRESSED = 0


class InterbankElement(Element):
    """Bank of the DebtRank cascade, with its level of distress, its state, and its economic value.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        h:
            The level of distress, between 0 and 1, where 1 is default.

        status:
            0 for undistressed, 1 for distressed, and 2 for inactive.

        value:
            The economic value, the bank's share of all interbank lending.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.finance import InterbankElement
        >>> GroupElement('banks', element=InterbankElement, size=3, values={'h': [1.0, 0.0, 0.0]}).size
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        h: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        status: Annotated[float, StateSpec(domain=NN)] = UNDISTRESSED,
        value: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.h = h
        self.status = status
        self.value = value

    @classmethod
    def _get_view(cls) -> type[InterbankView]:
        """Return the view of the agents of the element."""
        return InterbankView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since the cascade updates every bank at once.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []
