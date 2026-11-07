"""Value `x` of a map that keeps its last `delay` values, which a delayed coupling reads."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

from ....core.components.elements import DeclaredElement
from ....core.specs import DeclaredVariableSpec, StateSpec, VisualSpec


class DelayedUnitElement(DeclaredElement):
    """Value `x` of a map that keeps its last `delay` values, which a delayed coupling reads.

    The state `x` has `delay` lags, `x_lag1` to `x_lag<delay>`, and before the start every lag holds the initial value.

    Args:
        label:
            The label of the unit.

        visual:
            How the unit is drawn, or `None` for `default_visual_`.

        value:
            The initial value of `x`, also its value before the start.

        delay:
            The number of past values the unit keeps, at least 1.

        variables:
            The declared variables that replace the derived ones, or `None` for none.

    Attributes:
        default_visual_:
            The visual of the unit when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import DelayedUnitElement
        >>> DelayedUnitElement('unit', delay=2).get_value('x')
        0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        value: float = 0.3,
        delay: int = 1,
        variables: Mapping[str, DeclaredVariableSpec] | None = None,
    ) -> None:
        super().__init__(label, visual=visual, variables=variables)
        self.value = value
        self.delay = delay

    def _declare_variables(self) -> Mapping[str, DeclaredVariableSpec]:
        """Return the value `x` with its `delay` lags, all at the initial value, and the replacing variables."""
        lagged = StateSpec(lags=self.delay, initial=(self.value,) * self.delay)
        return {'x': DeclaredVariableSpec(self.value, lagged), **(self.variables or {})}
