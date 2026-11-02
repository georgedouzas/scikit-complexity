"""Firm of a Cournot duopoly, described by its output and its constant marginal cost."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec
from ._duopolist_view import DuopolistView


class DuopolistElement(Element):
    """Firm of a Cournot duopoly, described by its output and its constant marginal cost.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        output:
            The initial output.

        cost:
            The marginal cost.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.markets import DuopolistElement
        >>> GroupElement('firms', element=DuopolistElement, size=2, values={'output': [100.0, 100.0]}).size
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        output: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 50.0,
        cost: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.output = output
        self.cost = cost

    @classmethod
    def _get_view(cls) -> type[DuopolistView]:
        """Return the view of a duopolist, with its output and cost."""
        return DuopolistView
