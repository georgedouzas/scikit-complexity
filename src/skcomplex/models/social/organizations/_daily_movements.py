"""Movements of the women of the brothel movement model."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np

from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RuleLaw
from ....engine.solvers import ModelView
from ....errors import SpaceError
from ....specs import RoleSpec
from ._premises_view import PremisesView
from ._traveller_view import TravellerView


class DailyMovementsInteraction(Interaction):
    """Movements of the women of the brothel movement model.

    Every day one random free woman moves to a random cell of a random city, and in the steps of the phase
    `victims_phase` the traffickers move a random number of the victims of every brothel to random brothels. A move
    between cities counts as a train trip. The cities are vertical bands of equal width of the grid.

    Args:
        label:
            The label of the interaction.

        women:
            The path of the group of travellers.

        premises:
            The path of the group of premises, whose brothels the victims move between.

        n_cities:
            The number of cities, vertical bands of equal width.

        phase:
            The phase of the step the free woman moves in, or `None` for the default phase.

        victims_phase:
            The phase of the step the victims move in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.models.social.organizations import DailyMovementsInteraction
        >>> DailyMovementsInteraction('day', women='women', premises='premises').n_cities
        3
    """

    def __init__(
        self,
        label: str,
        *,
        women: Annotated[str, RoleSpec()],
        premises: Annotated[str, RoleSpec()],
        n_cities: int = 3,
        phase: str | None = None,
        victims_phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.women = women
        self.premises = premises
        self.n_cities = n_cities
        self.phase = phase
        self.victims_phase = victims_phase

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rules of the free woman and of the victims.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.move_free, phase=self.phase), RuleLaw(self.move_victims, phase=self.victims_phase)]

    def _city(self, location: object, width: int) -> int:
        """Return the city of a cell of a grid of a given width."""
        return DailyMovementsInteraction._cell(location)[0] * self.n_cities // width

    @staticmethod
    def _shape(model: ModelView) -> tuple[int, int]:
        """Return the width and the height of the grid of the model."""
        dimensions = model.dimensions
        if dimensions is None:
            error_msg = 'The trafficking model runs on a grid, and its agents live in no space.'
            raise SpaceError(error_msg)
        width, height = dimensions
        return int(width), int(height)

    def _travel(self, woman: TravellerView, cell: tuple[int, int], width: int) -> None:
        """Move a woman to a cell and count a trip when the city changes."""
        if self._city(cell, width) != self._city(woman.location, width):
            woman.trips += 1
        woman.move_to(cell)

    def move_free(self, model: ModelView) -> None:
        """Move a random free woman to a random cell of a random city.

        Args:
            model:
                The model.
        """
        free = [woman for woman in model.agents('women', TravellerView) if not woman.victim]
        if not free:
            return
        width, height = self._shape(model)
        city = int(model.rng.integers(self.n_cities))
        start, stop = city * width // self.n_cities, (city + 1) * width // self.n_cities
        cell = (int(model.rng.integers(start, stop)), int(model.rng.integers(height)))
        self._travel(free[int(model.rng.integers(len(free)))], cell, width)

    def move_victims(self, model: ModelView) -> None:
        """Move a random number of the victims of every brothel to random brothels.

        Args:
            model:
                The model.
        """
        women = model.agents('women', TravellerView)
        brothels = [
            DailyMovementsInteraction._cell(place.location)
            for place in model.agents('premises', PremisesView)
            if place.brothel
        ]
        width = self._shape(model)[0]
        for brothel in brothels:
            victims = [
                woman for woman in women if woman.victim and DailyMovementsInteraction._cell(woman.location) == brothel
            ]
            moved = model.rng.permutation(len(victims))[: int(model.rng.integers(len(victims) + 1))]
            for index in moved:
                self._travel(victims[index], brothels[int(model.rng.integers(len(brothels)))], width)

    @staticmethod
    def _cell(location: object) -> tuple[int, int]:
        """Return a grid location as a tuple of integers."""
        column, row = (int(item) for item in np.ravel(np.asarray(location)))
        return column, row
