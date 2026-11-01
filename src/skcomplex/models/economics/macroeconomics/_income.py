"""Economy described by its national income, which keeps the income of the previous period."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class IncomeElement(Element):
    """Economy described by its national income, which keeps the income of the previous period.

    Args:
        label:
            The label of the economy.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        Y:
            The income of the current period.

        Y_lag1:
            The income of the previous period.

        c:
            The marginal propensity to consume.

        v:
            The accelerator, the investment response to the change in consumption.

        G:
            The government spending.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import IncomeElement
        >>> IncomeElement('economy', c=0.5).c
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        Y: Annotated[object, StateSpec(lags=1, initial=(100.0,))] = 110.0,
        Y_lag1: float = 100.0,
        c: Annotated[object, ParameterSpec()] = 0.8,
        v: Annotated[object, ParameterSpec()] = 1.5,
        G: Annotated[object, ParameterSpec()] = 20.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.Y = Y
        self.Y_lag1 = Y_lag1
        self.c = c
        self.v = v
        self.G = G
