"""Agent of the minority game, who holds two strategies and plays the one it chose."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, ZZ
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec
from ._minority_player_view import MinorityPlayerView


class MinorityPlayerElement(Element):
    """Agent of the minority game, who holds two strategies and plays the one it chose.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        first:
            The code of the first strategy.

        second:
            The code of the second strategy.

        first_score:
            The virtual score of the first strategy, the number of times it would have been in the minority.

        second_score:
            The virtual score of the second strategy.

        strategy:
            The strategy played, 0 for the first and 1 for the second.

        action:
            The last action, `+1` or `-1`.

        wealth:
            The net number of times the agent was in the minority.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.econophysics import MinorityPlayerElement
        >>> GroupElement('players', element=MinorityPlayerElement, size=11, values={'first': 3, 'second': 5}).size
        11
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        first: Annotated[float, StateSpec(domain=NN)] = 0,
        second: Annotated[float, StateSpec(domain=NN)] = 1,
        first_score: Annotated[float, StateSpec()] = 0.0,
        second_score: Annotated[float, StateSpec()] = 0.0,
        strategy: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        action: Annotated[float, StateSpec(domain=ZZ)] = 0.0,
        wealth: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.first = first
        self.second = second
        self.first_score = first_score
        self.second_score = second_score
        self.strategy = strategy
        self.action = action
        self.wealth = wealth

    @classmethod
    def _get_view(cls) -> type[MinorityPlayerView]:
        """Return the view of the agents of the element."""
        return MinorityPlayerView
