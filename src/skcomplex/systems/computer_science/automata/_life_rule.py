"""Define the life-like rule of cells, such as `B3/S23` of Conway's Game of Life."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

from ....core.components.base import BaseComponent
from ....core.components.interactions import Interaction
from ....core.components.laws import LocalRuleLaw
from ....core.components.spaces import BaseNeighborhood, BaseSpace, RadiusNeighborhood
from ....core.specs import Role
from ._codes import life

MOORE = RadiusNeighborhood(1, metric='chebyshev')


class LifeRuleInteraction(Interaction):
    """Life-like rule of cells, such as `B3/S23` of Conway's Game of Life.

    A dead cell comes to life when its number of live neighbours is one of `born`, and a live cell stays alive when it
    is one of `survive`. Every other cell is dead at the next step.

    Args:
        label:
            The label of the interaction.

        target:
            The cells whose state `alive` the rule updates.

        born:
            The numbers of live neighbours that bring a dead cell to life.

        survive:
            The numbers of live neighbours that keep a live cell alive.

        neighborhood:
            The neighbours every cell counts, the eight cells around it by default.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import LifeRuleInteraction
        >>> LifeRuleInteraction('rule', target='cells', born=(3, 6)).born
        (3, 6)
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'cells',
        born: Sequence[int] = (3,),
        survive: Sequence[int] = (2, 3),
        neighborhood: BaseNeighborhood = MOORE,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.born = born
        self.survive = survive
        self.neighborhood = neighborhood

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the local rule of the cells as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            rule = life(self.born, self.survive)
            alive = context.target.alive
            self.laws_: list[BaseComponent] = [LocalRuleLaw(alive, self.neighborhood, rule, label=self.label)]
