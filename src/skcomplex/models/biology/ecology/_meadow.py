"""Ecology models: cows that graze a meadow of regrowing grass."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar, cast

import numpy as np

from ....core.base import Domain, VisualSpec, parameter, role, state
from ....core.model import AgentType, Interaction, LatticeSpace, Law, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView


def _cell(agent: AgentView) -> tuple[int, ...]:
    """Return the cell of an agent as a tuple of integers."""
    return tuple(int(item) for item in np.atleast_1d(cast('Any', agent.location)))


class Pasture(AgentType):
    """Patch of grass that regrows logistically every week, `G' = G + g G (1 - G)`.

    Args:
        label:
            The label of the agent type or group.

        location:
            The cell of the patch, or one cell per patch of a group.

        grass:
            The amount of grass, between 0 and 1.

        growth:
            The weekly growth rate `g` of the grass.

    Examples:
        >>> from skcomplex.models.biology.ecology import Pasture
        >>> Pasture.group('pasture', 4, growth=0.1).size
        4
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'grass', 'color_range': (0.0, 1.0)})
    grass: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    growth: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the regrowth rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.regrow)]

    def regrow(self, agent: AgentView, model: ModelView) -> None:
        """Grow the grass by one week of logistic growth.

        Args:
            agent:
                The patch.

            model:
                The model.
        """
        del model
        agent.grass += agent.growth * agent.grass * (1 - agent.grass)


class Cow(AgentType):
    """Cow of the meadow, which the grazing and herding interactions feed and move every week.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial cell, or one cell per cow of a group.

        bite:
            The grass a cow eats in a week.

    Examples:
        >>> from skcomplex.models.biology.ecology import Cow
        >>> Cow.group('cows', 10).size
        10
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'radius': 0.4})
    bite: float = parameter(default=0.75, domain=Domain.NONNEGATIVE)


class Grazing(Interaction):
    """Weekly meal of every cow, which eats its bite of the grass of the patch in its cell.

    Args:
        label:
            The label of the interaction.

        cows:
            The path of the group of cows.

        pasture:
            The path of the group of patches.

    Examples:
        >>> from skcomplex.models.biology.ecology import Grazing
        >>> Grazing('grazing', cows='cows', pasture='pasture').pasture
        'pasture'
    """

    cows: str = role()
    pasture: str = role()

    def laws(self, s: Symbols) -> list[Law]:
        """Return the grazing rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.graze)]

    def graze(self, model: ModelView) -> None:
        """Make every cow eat grass from the patch of its cell.

        Args:
            model:
                The model.
        """
        patches = {_cell(patch): patch for patch in model.agents('pasture')}
        for cow in model.agents('cows'):
            patch = patches.get(_cell(cow))
            if patch is not None:
                patch.grass = max(patch.grass - cow.bite, 0.0)


class Herding(Interaction):
    """Weekly move of every cow, in random order, to the nearest free patch with enough grass, or death.

    Args:
        label:
            The label of the interaction.

        cows:
            The path of the group of cows.

        pasture:
            The path of the group of patches.

        threshold:
            The least grass a patch needs for a cow to move there.

    Examples:
        >>> from skcomplex.models.biology.ecology import Herding
        >>> Herding('herding', cows='cows', pasture='pasture').threshold
        0.9
    """

    cows: str = role()
    pasture: str = role()
    threshold: float = parameter(default=0.9, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the herding rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.herd)]

    def herd(self, model: ModelView) -> None:
        """Move every cow to the nearest free patch with enough grass, and remove the cows that find none.

        Args:
            model:
                The model.
        """
        patches = model.agents('pasture')
        cells = [_cell(patch) for patch in patches]
        index = {cell: number for number, cell in enumerate(cells)}
        positions = np.asarray(cells, dtype=float)
        grown = np.asarray([patch.grass for patch in patches]) >= model.parameter('threshold')
        cows = model.agents('cows')
        occupied = np.zeros(len(cells), dtype=bool)
        occupied[[index[_cell(cow)] for cow in cows]] = True
        for number in model.rng.permutation(len(cows)):
            cow = cows[number]
            occupied[index[_cell(cow)]] = False
            free = grown & ~occupied
            if not free.any():
                cow.remove()
                continue
            distances = np.hypot(*(positions - np.asarray(_cell(cow), dtype=float)).T)
            target = int(np.argmin(np.where(free, distances, np.inf)))
            cow.move_to(cells[target])
            occupied[target] = True


@register_model(
    field='biology/ecology',
    tags=('carrying capacity', 'agents', 'grid', 'grazing', 'commons'),
    references=('Hamill and Gilbert 2016',),
    defaults={'until': 25},
)
def meadow_capacity(
    width: int = 40,
    height: int = 40,
    *,
    cows: int = 80,
    growth: float = 0.2,
    bite: float = 0.75,
    threshold: float = 0.9,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return cows that graze a summer of 25 weeks on a meadow, which shows how many cows the meadow can carry.

    Every week the grass regrows logistically, each cow eats `bite` of the grass of its patch, and each cow moves to the
    nearest patch that no other cow occupies and whose grass reaches `threshold`. A cow that finds none dies. Without
    regrowth a cow grazes its first patch and 25 more over the summer, so the meadow feeds about
    `width * height / 26` cows, and regrowth raises its carrying capacity.
    The cows graze the patches down at the same pace, so a herd above the capacity runs out of grown patches all at
    once and dies out within a week, instead of shrinking to the capacity.

    Args:
        width:
            The number of columns of the meadow.

        height:
            The number of rows of the meadow.

        cows:
            The initial number of cows.

        growth:
            The weekly growth rate of the grass.

        bite:
            The grass a cow eats in a week.

        threshold:
            The least grass a patch needs for a cow to move there.

        random_state:
            The seed or generator of the initial cells of the cows.

    Returns:
        system:
            The model, with the groups `pasture` and `cows`.

    Examples:
        >>> from skcomplex.models.biology.ecology import meadow_capacity
        >>> [group.size for group in meadow_capacity(10, 10, cows=5).elements]
        [100, 5]
    """
    rng = np.random.default_rng(random_state)
    cells = [(column, row) for column in range(width) for row in range(height)]
    picks = rng.choice(len(cells), size=cows, replace=False)
    return System(
        components=[
            Pasture.group('pasture', len(cells), location=cells, growth=growth),
            Cow.group('cows', cows, location=[cells[int(pick)] for pick in picks], bite=bite),
        ],
        interactions=[
            Grazing('grazing', cows='cows', pasture='pasture'),
            Herding('herding', cows='cows', pasture='pasture', threshold=threshold),
        ],
        space=LatticeSpace((width, height), boundary='fixed'),
    )
