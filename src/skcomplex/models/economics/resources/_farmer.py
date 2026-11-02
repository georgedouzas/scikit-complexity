"""Farmer who adds the catch of every step to its wealth, reading the resource from the other partition."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RuleLaw
from ....engine.solvers import ModelView
from ....errors import ValidationError
from ....specs import StateSpec, VisualSpec
from ._farmer_view import FarmerView


class FarmerElement(Element):
    """Farmer who adds the catch of every step to its wealth, reading the resource from the other partition.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        wealth:
            The initial wealth.

        resource:
            The dotted name of the resource stock.

        effort:
            The dotted name of the harvest effort per farmer.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.resources import FarmerElement
        >>> GroupElement('farmers', element=FarmerElement, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        wealth: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        resource: str = 'land/stock.n',
        effort: str = 'harvest.effort',
    ) -> None:
        super().__init__(label, visual=visual)
        self.wealth = wealth
        self.resource = resource
        self.effort = effort

    @classmethod
    def _get_view(cls) -> type[FarmerView]:
        """Return the view of the agents of the element."""
        return FarmerView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the catch rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.catch)]

    def catch(self, agent: FarmerView, model: ModelView) -> None:
        """Add the effort times the resource stock at the last synchronisation point to the wealth.

        Args:
            agent:
                The farmer.

            model:
                The model.
        """
        stock = model.inputs.get(self.resource, 0.0)
        if not isinstance(stock, int | float | np.number):
            error_msg = f'The input {self.resource!r} of the farmers is {stock!r}, not a number.'
            raise ValidationError(error_msg)
        agent.wealth += model.parameters[self.effort] * float(stock)
