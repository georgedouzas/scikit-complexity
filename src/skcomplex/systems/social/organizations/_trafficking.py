"""Trafficking systems: victims moved between brothels and the traces they leave.

The traces are left by their movements.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, cast, runtime_checkable

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.compile import PhaseSpec
from ....core.engine.solvers import BaseSolver, MesaSolver, SystemView
from ....core.errors import SpaceError
from ....core.systems import BaseSystem, System
from ....core.types import Role

MONTH = 30
QUARTER = 90
DAILY = 'default'
MONTHLY = 'trafficking'
BILLING = 'billing'
QUARTERLY = 'quarter'


@runtime_checkable
class PremisesView(AgentViewProtocol, Protocol):
    """View of a premises of the trafficking system.

    Its states and parameters are attributes.
    """

    brothel: float
    bills: float
    last_bills: float
    efficiency: float
    day: float


class PremisesElement(Element):
    """House or brothel of the brothel movement system.

    Its energy bills grow with the people inside. Every day the bills grow by `1 / (1 + exp(x0 - n))` for the `n` people
    in the premises' cell, and every quarter they are recorded and reset. In `TraffickingSystem`, maps of the bills read
    the counts of the women by cell every day, and maps in a phase of a period of 90 days record and reset them.

    Args:
        label:
            The label of the agent or group.

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

        values:
            Further values by name, such as the cell `x` and `y` of every member.

        size:
            The number of members, or `None` for one.

        space:
            The grid whose axes `x` and `y` place the members, or `None` for none.

    Examples:
        >>> from skcomplex.systems.social.organizations import PremisesElement
        >>> group = PremisesElement('premises', brothel=1.0, size=3)
        >>> group.label, group.size
        ('premises', 3)
    """

    visual = VisualSpec('agents', {'color_by': 'brothel', 'color_range': (0.0, 1.0)})

    def __init__(
        self,
        label: str,
        *,
        brothel: Annotated[object, FiniteEnumeratedSet([0, 1])] = 0.0,
        bills: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        last_bills: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        efficiency: object = 1.0,
        day: Annotated[object, NN] = 0.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.brothel = brothel
        self.bills = bills
        self.last_bills = last_bills
        self.efficiency = efficiency
        self.day = day


class TravellerView(AgentViewProtocol, Protocol):
    """View of a woman of the trafficking system.

    Its states and parameters are attributes.
    """

    victim: float
    trips: float


class TravellerElement(Element):
    """Woman of the brothel movement system.

    She is a trafficking victim, or a woman who travels freely.

    Args:
        label:
            The label of the agent or group.

        victim:
            Whether the woman is a trafficking victim, 1, or not, 0.

        trips:
            The number of train trips between cities.

        values:
            Further values by name, such as the cell `x` and `y` of every member.

        size:
            The number of members, or `None` for one.

        space:
            The grid whose axes `x` and `y` place the members, or `None` for none.

    Examples:
        >>> from skcomplex.systems.social.organizations import TravellerElement
        >>> group = TravellerElement('women', victim=1.0, size=4)
        >>> group.label, group.size
        ('women', 4)
    """

    visual = VisualSpec('agents', {'color_by': 'victim', 'color_range': (0.0, 1.0)})

    def __init__(
        self,
        label: str,
        *,
        victim: Annotated[object, FiniteEnumeratedSet([0, 1])] = 0.0,
        trips: Annotated[object, NN] = 0.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.victim = victim
        self.trips = trips


class DailyMovementsInteraction(Interaction):
    """Movements of the women of the brothel movement system.

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
        >>> from skcomplex.systems.social.organizations import DailyMovementsInteraction
        >>> DailyMovementsInteraction('day', women='women', premises='premises').n_cities
        3
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        women: Role,
        premises: Role,
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

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rules of the free women and of the victims as rules of the system, in their phases."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = [
                UpdateLaw(context, self.move_free, phase=self.phase),
                UpdateLaw(context, self.move_victims, phase=self.victims_phase),
            ]

    def _get_city(self, location: object, width: int) -> int:
        """Return the city of a cell of a grid of a given width."""
        return DailyMovementsInteraction._get_cell(location)[0] * self.n_cities // width

    @staticmethod
    def _get_shape(system: SystemView) -> tuple[int, int]:
        """Return the width and the height of the grid of the system."""
        dimensions = system.dimensions
        if dimensions is None:
            error_msg = 'The trafficking system runs on a grid, and its agents live in no space.'
            raise SpaceError(error_msg)
        width, height = dimensions
        return int(width), int(height)

    def _travel(self, woman: TravellerView, cell: tuple[int, int], width: int) -> None:
        """Move a woman to a cell and count a trip when the city changes."""
        if self._get_city(cell, width) != self._get_city(woman.location, width):
            woman.trips += 1
        woman.move_to(cell)

    def move_free(self, system: SystemView) -> None:
        """Move a random free woman to a random cell of a random city.

        Args:
            system:
                The system.
        """
        free = [woman for woman in cast(list[TravellerView], system.get_agents('women')) if not woman.victim]
        if not free:
            return
        width, height = self._get_shape(system)
        city = int(system.rng.integers(self.n_cities))
        start, stop = city * width // self.n_cities, (city + 1) * width // self.n_cities
        cell = (int(system.rng.integers(start, stop)), int(system.rng.integers(height)))
        self._travel(free[int(system.rng.integers(len(free)))], cell, width)

    def move_victims(self, system: SystemView) -> None:
        """Move a random number of the victims of every brothel to random brothels.

        Args:
            system:
                The system.
        """
        women = cast(list[TravellerView], system.get_agents('women'))
        brothels = [
            DailyMovementsInteraction._get_cell(place.location)
            for place in cast(list[PremisesView], system.get_agents('premises'))
            if place.brothel
        ]
        width = self._get_shape(system)[0]
        for brothel in brothels:
            victims = [
                woman
                for woman in women
                if woman.victim and DailyMovementsInteraction._get_cell(woman.location) == brothel
            ]
            moved = system.rng.permutation(len(victims))[: int(system.rng.integers(len(victims) + 1))]
            for index in moved:
                self._travel(victims[index], brothels[int(system.rng.integers(len(brothels)))], width)

    @staticmethod
    def _get_cell(location: object) -> tuple[int, int]:
        """Return a grid location as a tuple of integers."""
        column, row = (int(item) for item in np.ravel(np.asarray(location)))
        return column, row


class TraffickingSystem(BaseSystem):
    """The brothel movement system.

    Victims moved between brothels leave traces in bills and train trips. The system has the groups `premises` and
    `women`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.organizations import TraffickingSystem
        >>> system = TraffickingSystem(n_houses=5, n_brothels=2, n_victims=4, n_free=3).prepare().system_
        >>> [group.size for group in system.elements]
        [7, 7]
    """

    field = 'social/organizations'
    topics = ('trafficking', 'agents', 'grid', 'crime')
    references = ('Chesney 2021',)
    until = 90

    def __init__(
        self,
        width: int = 30,
        height: int = 30,
        *,
        n_cities: int = 3,
        n_houses: int = 40,
        n_brothels: int = 3,
        n_victims: int = 12,
        n_free: int = 30,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.width = width
        self.height = height
        self.n_cities = n_cities
        self.n_houses = n_houses
        self.n_brothels = n_brothels
        self.n_victims = n_victims
        self.n_free = n_free
        self.random_state = random_state

    @staticmethod
    def _add_bills(system: SystemViewProtocol) -> None:
        """Count a day at every premises and add to its bills the growth `1 / (1 + exp(x0 - n))` for the `n` inside."""
        places = [place for place in system.get_agents('premises') if isinstance(place, PremisesView)]
        for place in places:
            place.day = place.day + 1.0
        occupancy = system.get_occupancy('women')
        heat = [float(1 / (1 + np.exp(place.efficiency - occupancy.get(place.location_key, 0)))) for place in places]
        for place, growth in zip(places, heat, strict=True):
            place.bills = place.bills + growth

    @staticmethod
    def _close_quarter(system: SystemViewProtocol) -> None:
        """Record the bills of the quarter that ends at every premises and start the next quarter at zero."""
        places = [place for place in system.get_agents('premises') if isinstance(place, PremisesView)]
        for place in places:
            place.last_bills = float(place.bills)
        for place in places:
            place.bills = 0.0

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        n_places = self.n_houses + self.n_brothels
        cells = rng.permutation(self.width * self.height)[:n_places]
        places = [(int(cell % self.width), int(cell // self.width)) for cell in cells]
        brothel = (np.arange(n_places) >= self.n_houses).astype(float)
        space = LatticeSpace((self.width, self.height), boundary='fixed')
        x, y = np.array(places).T
        premises = PremisesElement(
            'premises',
            brothel=brothel,
            efficiency=rng.uniform(0, 3, n_places),
            values={'x': x, 'y': y},
            size=n_places,
            space=space,
        )
        homes = [places[self.n_houses + index % self.n_brothels] for index in range(self.n_victims)]
        homes += [(int(rng.integers(self.width)), int(rng.integers(self.height))) for _ in range(self.n_free)]
        victim = (np.arange(self.n_victims + self.n_free) < self.n_victims).astype(float)
        x, y = np.array(homes).T
        women = TravellerElement(
            'women',
            victim=victim,
            values={'x': x, 'y': y},
            size=self.n_victims + self.n_free,
            space=space,
        )
        day = DailyMovementsInteraction(
            'day',
            women='women',
            premises='premises',
            n_cities=self.n_cities,
            phase=DAILY,
            victims_phase=MONTHLY,
        )
        bills = [
            Interaction(
                'bills',
                roles={'premises': 'premises', 'women': 'women'},
                laws=lambda s: [
                    UpdateLaw(s, self._add_bills, phase=BILLING),
                    UpdateLaw(s, self._close_quarter, phase=QUARTERLY),
                ],
            ),
        ]
        phases = [
            PhaseSpec(DAILY, activation='sequential'),
            PhaseSpec(MONTHLY, activation='sequential', period=MONTH),
            PhaseSpec(BILLING, activation='sequential'),
            PhaseSpec(QUARTERLY, activation='sequential', period=QUARTER),
        ]
        self.system_ = System(
            self.label,
            elements=[premises, women],
            interactions=[day, *bills],
            space=space,
            solver=MesaSolver(phases=phases),
        )
