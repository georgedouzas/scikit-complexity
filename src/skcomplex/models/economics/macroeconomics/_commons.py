"""Harvest of the hunter-gatherers, who pool the labour of the adults on a patch and share the food equally."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RuleLaw
from ....engine.solvers import ModelView
from ....specs import (
    RoleSpec,
)
from ._settler import SettlerElement
from ._settler_view import SettlerView


class CommonsInteraction(Interaction):
    """Harvest of the hunter-gatherers, who pool the labour of the adults on a patch and share the food equally.

    The adults of a patch harvest `min(food, labour - resistance)`, and everyone on the patch, child or adult, gets an
    equal share.

    Args:
        label:
            The label of the interaction.

        people:
            The path of the group of people.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import CommonsInteraction
        >>> CommonsInteraction('commons', people='people').people
        'people'
    """

    def __init__(
        self,
        label: str,
        *,
        people: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.people = people

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the harvest rule in the phase of work.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.harvest, phase='work')]

    def harvest(self, model: ModelView) -> None:
        """Harvest every occupied patch and share the food among the people on it.

        Args:
            model:
                The model view of the interaction.
        """
        cells: dict[tuple[int, ...], list[SettlerView]] = {}
        for person in model.agents('people', SettlerView):
            cells.setdefault(SettlerElement._cell(person), []).append(person)
        for group in cells.values():
            patch = SettlerElement._patch_here(group[0])
            labour = sum(person.labour for person in group if person.age >= person.maturity)
            harvest = min(patch.food, max(labour - patch.resistance, 0.0))
            patch.food -= harvest
            for person in group:
                person.food += harvest / len(group)
                person.harvested = harvest / len(group)
