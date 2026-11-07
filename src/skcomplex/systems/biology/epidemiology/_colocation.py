"""Daily movement of people between their homes and workplaces, and infection between people at one location."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import SystemView
from ....core.specs import RoleSpec
from ._commuter_view import CommuterView

SUSCEPTIBLE, INFECTED, RECOVERED = 0.0, 1.0, 2.0


class ColocationInteraction(Interaction):
    """Daily movement of people between their homes and workplaces, and infection between people at one location.

    A step is half a day: in even steps every person is at home and in odd steps at the workplace, unless workplaces
    are closed, the time-labelled visits of the bipartite graph of people and locations of EpiSims. At every step a
    susceptible person is infected with the probability `1 - (1 - p)^k` by the `k` infected people at the same location,
    and an infected person recovers with the probability `gamma`.

    Args:
        label:
            The label of the interaction.

        people:
            The path of the agent group of people.

        p:
            The probability of transmission per infected contact and step.

        gamma:
            The probability of recovery per step.

        closed:
            Whether the workplaces are closed, so people stay at home.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import ColocationInteraction
        >>> ColocationInteraction('contacts', people='people', p=0.02).p
        0.02
    """

    def __init__(
        self,
        label: str,
        *,
        people: Annotated[str, RoleSpec()],
        p: float = 0.05,
        gamma: float = 0.1,
        closed: bool = False,
    ) -> None:
        super().__init__(label)
        self.people = people
        self.p = p
        self.gamma = gamma
        self.closed = closed

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of a half day.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.visit)]

    def visit(self, system: SystemView) -> None:
        """Move every person to the location of the half day, then infect and recover.

        Args:
            system:
                The system.
        """
        people = system.agents(self.people, CommuterView)
        at_work = round(system.time) % 2 == 1 and not self.closed
        places = np.array([int(person.work if at_work else person.home) for person in people])
        health = np.array([person.health for person in people])
        infected = np.bincount(places[health == INFECTED], minlength=places.max() + 1)[places]
        draws = system.rng.random(len(people))
        infection = (health == SUSCEPTIBLE) & (draws < 1 - (1 - self.p) ** infected)
        recovery = (health == INFECTED) & (draws < self.gamma)
        health = np.where(infection, INFECTED, np.where(recovery, RECOVERED, health))
        for person, place, state in zip(people, places, health, strict=True):
            person.move_to(int(place))
            person.health = float(state)
