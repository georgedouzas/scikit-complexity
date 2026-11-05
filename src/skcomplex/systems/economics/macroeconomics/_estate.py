"""Settlement of the estates of the dead, who leave their money and loans to an heir, with a census of the step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.laws.utils import transfer
from ....core.engine.solvers import SystemView
from ....core.specs import (
    RoleSpec,
)
from ._settler import AGED, ALIVE, STARVED
from ._settler_view import SettlerView


class EstateInteraction(Interaction):
    """Settlement of the estates of the dead, who leave their money and loans to an heir, with a census of the step.

    The money and the loans of a dead person go to a random living person of its patch, or of anywhere. When nobody
    lives to inherit them, the dead person stays as an estate that holds them. The reporters `births`, `starvations`,
    and `old_age` count the events of a step.

    Args:
        label:
            The label of the interaction.

        people:
            The path of the group of people.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import EstateInteraction
        >>> EstateInteraction('census', people='people').reports
        ('births', 'starvations', 'old_age')
    """

    reports: ClassVar[tuple[str, ...]] = ('births', 'starvations', 'old_age')

    def __init__(
        self,
        label: str,
        *,
        people: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.people = people

    def _settle(self, dead: SettlerView, living: list[SettlerView], system: SystemView) -> None:
        """Leave the money and the loans of a dead person to an heir, and remove the person."""
        near = [person for person in living if person.location == dead.location] or living
        if near:
            heir = near[int(system.rng.integers(len(near)))]
            transfer(dead, heir, dead.money, 'money')
            transfer(dead, heir, dead.loans, 'loans')
        if dead.money == 0 and dead.loans == 0:
            dead.remove()

    def count(self, system: SystemView) -> None:
        """Count the births and deaths of the step and remove the dead.

        Args:
            system:
                The system view of the interaction.
        """
        people = system.agents('people', SettlerView)
        living = [person for person in people if person.fate == ALIVE]
        dead = [person for person in people if person.fate != ALIVE]
        system.report('births', float(sum(person.age == 0 for person in living)))
        system.report('starvations', float(sum(person.fate == STARVED for person in dead)))
        system.report('old_age', float(sum(person.fate == AGED for person in dead)))
        for person in dead:
            self._settle(person, living, system)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the counting rule in the phase of the census.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.count, phase='census')]
