"""Strategy of Robby, the action for each of the 243 situations he can see, with its last measured score."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec
from ._strategy_view import StrategyView


class StrategyElement(Element):
    """Strategy of Robby, the action for each of the 243 situations he can see, with its last measured score.

    The actions are 0 to 3 for moving north, south, east, and west, 4 to stay, 5 to pick up a can, and 6 to move in a
    random direction.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        actions:
            The initial action per situation, an array of 243 integers from 0 to 6.

        score:
            The initial score.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.computer_science.evolutionary_computation import StrategyElement
        >>> GroupElement('strategies', element=StrategyElement, size=3).get_params()['size']
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'score'}),
        *,
        actions: Annotated[object, StateSpec()] = 0.0,
        score: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.actions = actions
        self.score = score

    @classmethod
    def _get_view(cls) -> type[StrategyView]:
        """Return the view of the strategies, which types their states."""
        return StrategyView
