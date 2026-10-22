"""Trafficking models: victims moved between brothels and the traces their movements leave."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar, cast

import numpy as np

from ....core.base import Domain, VisualSpec, role, state
from ....core.model import AgentType, Interaction, LatticeSpace, Law, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

MONTH = 30
QUARTER = 90


def _cell(location: object) -> tuple[int, int]:
    """Return a grid location as a tuple of integers."""
    column, row = (int(item) for item in np.ravel(np.asarray(location)))
    return column, row


class Premises(AgentType):
    """House or brothel of the brothel movement model, whose energy bills grow with the people inside.

    Every day the bills grow by `1 / (1 + exp(x0 - n))` for the `n` people in the premises' cell, and every quarter
    they are recorded and reset.

    Args:
        label:
            The label of the agent type or group.

        location:
            The cell, or `None` for a random cell.

        brothel:
            Whether the premises are a brothel, 1, or a house, 0.

        bills:
            The bills of the current quarter.

        last_bills:
            The bills of the last full quarter.

        efficiency:
            The midpoint `x0` of the sigmoid of the bills.

        day:
            The number of days run.

    Examples:
        >>> from skcomplex.models.social.organizations import Premises
        >>> Premises.group('premises', 3, brothel=1.0)
        Group('premises', element=Premises, size=3)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'brothel', 'color_range': (0.0, 1.0)})
    brothel: float = state(default=0.0, domain=Domain.BOOLEAN)
    bills: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    last_bills: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    efficiency: float = state(default=1.0)
    day: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)


class Traveller(AgentType):
    """Woman of the brothel movement model: a trafficking victim, or a woman who travels freely.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial cell, or `None` for a random cell.

        victim:
            Whether the woman is a trafficking victim, 1, or not, 0.

        trips:
            The number of train trips between cities.

    Examples:
        >>> from skcomplex.models.social.organizations import Traveller
        >>> Traveller.group('women', 4, victim=1.0)
        Group('women', element=Traveller, size=4)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'victim', 'color_range': (0.0, 1.0)})
    victim: float = state(default=0.0, domain=Domain.BOOLEAN)
    trips: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)


class DailyMovements(Interaction):
    """Day of the brothel movement model.

    One random free woman moves to a random cell of a random city. Every month the traffickers move a random number
    of the victims of every brothel to random brothels. A move between cities counts as a train trip. Then the bills
    of every premises grow, and every quarter they are recorded and reset. The cities are vertical bands of equal
    width of the grid.

    Args:
        label:
            The label of the interaction.

        women:
            The path of the group of travellers.

        premises:
            The path of the group of premises, which the rule reads and updates.

        n_cities:
            The number of cities, vertical bands of equal width.

    Examples:
        >>> from skcomplex.models.social.organizations import DailyMovements
        >>> DailyMovements('day', women='women', premises='premises').n_cities
        3
    """

    women: str = role()
    premises: str = role()
    n_cities: int = 3

    def laws(self, s: Symbols) -> list[Law]:
        """Return the daily rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.pass_day)]

    def _city(self, location: object, width: int) -> int:
        """Return the city of a cell of a grid of a given width."""
        return _cell(location)[0] * self.n_cities // width

    def _travel(self, woman: AgentView, cell: tuple[int, int], width: int) -> None:
        """Move a woman to a cell and count a trip when the city changes."""
        if self._city(cell, width) != self._city(woman.location, width):
            woman.trips += 1
        woman.move_to(cell)

    def _move_free(self, women: list[AgentView], model: ModelView) -> None:
        """Move a random free woman to a random cell of a random city."""
        free = [woman for woman in women if not woman.victim]
        if not free:
            return
        width, height = cast('tuple[int, int]', model.dimensions)
        city = int(model.rng.integers(self.n_cities))
        start, stop = city * width // self.n_cities, (city + 1) * width // self.n_cities
        cell = (int(model.rng.integers(start, stop)), int(model.rng.integers(height)))
        self._travel(free[int(model.rng.integers(len(free)))], cell, width)

    def _move_victims(self, women: list[AgentView], brothels: list[tuple[int, int]], model: ModelView) -> None:
        """Move a random number of the victims of every brothel to random brothels."""
        width = cast('tuple[int, int]', model.dimensions)[0]
        for brothel in brothels:
            victims = [woman for woman in women if woman.victim and _cell(woman.location) == brothel]
            moved = model.rng.permutation(len(victims))[: int(model.rng.integers(len(victims) + 1))]
            for index in moved:
                self._travel(victims[index], brothels[int(model.rng.integers(len(brothels)))], width)

    def pass_day(self, model: ModelView) -> None:
        """Move the women of one day and update the bills.

        Args:
            model:
                The model.
        """
        women, premises = model.agents('women'), model.agents('premises')
        if not premises:
            return
        day = premises[0].day + 1
        self._move_free(women, model)
        if day % MONTH == 0:
            self._move_victims(women, [_cell(place.location) for place in premises if place.brothel], model)
        counts: dict[tuple[int, int], int] = {}
        for woman in women:
            counts[_cell(woman.location)] = counts.get(_cell(woman.location), 0) + 1
        for place in premises:
            place.day = day
            place.bills += 1 / (1 + np.exp(place.efficiency - counts.get(_cell(place.location), 0)))
            if day % QUARTER == 0:
                place.last_bills, place.bills = place.bills, 0.0


@register_model(
    field='social/organizations',
    tags=('trafficking', 'agents', 'grid', 'crime'),
    references=('Chesney 2021',),
    defaults={'until': 90},
)
def trafficking(
    width: int = 30,
    height: int = 30,
    *,
    n_cities: int = 3,
    n_houses: int = 40,
    n_brothels: int = 3,
    n_victims: int = 12,
    n_free: int = 30,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the brothel movement model, where victims moved between brothels leave traces in bills and train trips.

    Args:
        width:
            The number of columns of the grid.

        height:
            The number of rows of the grid.

        n_cities:
            The number of cities, vertical bands of equal width.

        n_houses:
            The number of houses.

        n_brothels:
            The number of brothels.

        n_victims:
            The number of victims, spread over the brothels.

        n_free:
            The number of women who travel freely.

        random_state:
            The seed or generator of the places, the women, and the efficiencies.

    Returns:
        system:
            The model, with the groups `premises` and `women`.

    Examples:
        >>> from skcomplex.models.social.organizations import trafficking
        >>> [group.size for group in trafficking(n_houses=5, n_brothels=2, n_victims=4, n_free=3).elements]
        [7, 7]
    """
    rng = np.random.default_rng(random_state)
    n_places = n_houses + n_brothels
    cells = rng.permutation(width * height)[:n_places]
    places = [(int(cell % width), int(cell // width)) for cell in cells]
    brothel = (np.arange(n_places) >= n_houses).astype(float)
    premises = Premises.group(
        'premises',
        n_places,
        location=places,
        brothel=brothel,
        efficiency=rng.uniform(0, 3, n_places),
    )
    homes = [places[n_houses + index % n_brothels] for index in range(n_victims)]
    homes += [(int(rng.integers(width)), int(rng.integers(height))) for _ in range(n_free)]
    victim = (np.arange(n_victims + n_free) < n_victims).astype(float)
    women = Traveller.group('women', n_victims + n_free, location=homes, victim=victim)
    day = DailyMovements('day', women='women', premises='premises', n_cities=n_cities)
    return System(
        components=[premises, women], interactions=[day], space=LatticeSpace((width, height), boundary='fixed'),
    )
