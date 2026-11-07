"""Trafficking systems: victims moved between brothels and the traces their movements leave."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction, MapInteraction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.spaces import DiscreteTime, LatticeSpace
from ....core.components.system import BaseSystem, System
from ....core.components.types import SystemViewProtocol
from ....core.engine.solvers import AgentView, SystemView
from ....core.errors import SpaceError
from ....core.specs import PhaseSpec, PhasesScheduleSpec, RoleSpec, StateSpec, VisualSpec


class PremisesView(AgentView):
    """View of a premises of the trafficking system, with its states and parameters as attributes."""

    brothel: float
    bills: float
    last_bills: float
    efficiency: float
    day: float


class PremisesElement(Element):
    """House or brothel of the brothel movement system, whose energy bills grow with the people inside.

    Every day the bills grow by `1 / (1 + exp(x0 - n))` for the `n` people in the premises' cell, and every quarter they
    are recorded and reset. In `TraffickingSystem`, maps of the bills read the counts of the women by cell every day,
    and maps in a phase of a period of 90 days record and reset them.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.organizations import PremisesElement
        >>> group = GroupElement('premises', element=PremisesElement, size=3, values={'brothel': 1.0})
        >>> group.label, group.element.__name__, group.size
        ('premises', 'PremisesElement', 3)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'brothel', 'color_range': (0.0, 1.0)}),
        *,
        brothel: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        bills: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        last_bills: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        efficiency: Annotated[float, StateSpec()] = 1.0,
        day: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.brothel = brothel
        self.bills = bills
        self.last_bills = last_bills
        self.efficiency = efficiency
        self.day = day

    @classmethod
    def _get_view(cls) -> type[PremisesView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return PremisesView


class TravellerView(AgentView):
    """View of a woman of the trafficking system, with its states and parameters as attributes."""

    victim: float
    trips: float


class TravellerElement(Element):
    """Woman of the brothel movement system: a trafficking victim, or a woman who travels freely.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        victim:
            Whether the woman is a trafficking victim, 1, or not, 0.

        trips:
            The number of train trips between cities.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.organizations import TravellerElement
        >>> group = GroupElement('women', element=TravellerElement, size=4, values={'victim': 1.0})
        >>> group.label, group.element.__name__, group.size
        ('women', 'TravellerElement', 4)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'victim', 'color_range': (0.0, 1.0)}),
        *,
        victim: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        trips: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.victim = victim
        self.trips = trips

    @classmethod
    def _get_view(cls) -> type[TravellerView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return TravellerView


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

    law_classes = (RuleLaw,)

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
    def _shape(system: SystemView) -> tuple[int, int]:
        """Return the width and the height of the grid of the system."""
        dimensions = system.dimensions
        if dimensions is None:
            error_msg = 'The trafficking system runs on a grid, and its agents live in no space.'
            raise SpaceError(error_msg)
        width, height = dimensions
        return int(width), int(height)

    def _travel(self, woman: TravellerView, cell: tuple[int, int], width: int) -> None:
        """Move a woman to a cell and count a trip when the city changes."""
        if self._city(cell, width) != self._city(woman.location, width):
            woman.trips += 1
        woman.move_to(cell)

    def move_free(self, system: SystemView) -> None:
        """Move a random free woman to a random cell of a random city.

        Args:
            system:
                The system.
        """
        free = [woman for woman in system.agents('women', TravellerView) if not woman.victim]
        if not free:
            return
        width, height = self._shape(system)
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
        women = system.agents('women', TravellerView)
        brothels = [
            DailyMovementsInteraction._cell(place.location)
            for place in system.agents('premises', PremisesView)
            if place.brothel
        ]
        width = self._shape(system)[0]
        for brothel in brothels:
            victims = [
                woman for woman in women if woman.victim and DailyMovementsInteraction._cell(woman.location) == brothel
            ]
            moved = system.rng.permutation(len(victims))[: int(system.rng.integers(len(victims) + 1))]
            for index in moved:
                self._travel(victims[index], brothels[int(system.rng.integers(len(brothels)))], width)

    @staticmethod
    def _cell(location: object) -> tuple[int, int]:
        """Return a grid location as a tuple of integers."""
        column, row = (int(item) for item in np.ravel(np.asarray(location)))
        return column, row


MONTH = 30


QUARTER = 90


DAILY = 'default'


MONTHLY = 'trafficking'


BILLING = 'billing'


QUARTERLY = 'quarter'


class TraffickingSystem(BaseSystem):
    """The brothel movement system, where victims moved between brothels leave traces in bills and train trips.

    The system has the groups `premises` and `women`.

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
    ) -> None:
        super().__init__(label)
        self.width = width
        self.height = height
        self.n_cities = n_cities
        self.n_houses = n_houses
        self.n_brothels = n_brothels
        self.n_victims = n_victims
        self.n_free = n_free
        self.random_state = random_state

    @staticmethod
    def _heat(place: PremisesView, system: SystemViewProtocol) -> float:
        """Return the growth of the bills of a premises in a day, `1 / (1 + exp(x0 - n))` for the `n` people inside."""
        people = system.occupancy('sources').get(place.location_key, 0)
        return float(1 / (1 + np.exp(place.efficiency - people)))

    @staticmethod
    def _count_day(place: PremisesView, system: SystemViewProtocol) -> float:
        """Return one day."""
        return 1.0

    @staticmethod
    def _record(place: PremisesView, system: SystemViewProtocol) -> float:
        """Return the bills of the quarter that ends."""
        return float(place.bills)

    @staticmethod
    def _reset(place: PremisesView, system: SystemViewProtocol) -> float:
        """Return the bills at the start of a quarter."""
        return 0.0

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        n_places = self.n_houses + self.n_brothels
        cells = rng.permutation(self.width * self.height)[:n_places]
        places = [(int(cell % self.width), int(cell // self.width)) for cell in cells]
        brothel = (np.arange(n_places) >= self.n_houses).astype(float)
        premises = GroupElement(
            'premises',
            element=PremisesElement,
            size=n_places,
            values={'brothel': brothel, 'efficiency': rng.uniform(0, 3, n_places)},
            location=places,
        )
        homes = [places[self.n_houses + index % self.n_brothels] for index in range(self.n_victims)]
        homes += [(int(rng.integers(self.width)), int(rng.integers(self.height))) for _ in range(self.n_free)]
        victim = (np.arange(self.n_victims + self.n_free) < self.n_victims).astype(float)
        women = GroupElement(
            'women',
            element=TravellerElement,
            size=self.n_victims + self.n_free,
            values={'victim': victim},
            location=homes,
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
            MapInteraction(
                'days',
                target='premises',
                state='day',
                function=self._count_day,
                mode='increment',
                form='agent',
                phase=BILLING,
            ),
            MapInteraction(
                'bills',
                target='premises',
                state='bills',
                function=self._heat,
                sources=['women'],
                mode='increment',
                form='agent',
                phase=BILLING,
            ),
            MapInteraction(
                'record',
                target='premises',
                state='last_bills',
                function=self._record,
                form='agent',
                phase=QUARTERLY,
            ),
            MapInteraction(
                'reset',
                target='premises',
                state='bills',
                function=self._reset,
                form='agent',
                phase=QUARTERLY,
            ),
        ]
        phases = [
            PhaseSpec(DAILY, activation='sequential'),
            PhaseSpec(MONTHLY, activation='sequential', period=MONTH),
            PhaseSpec(BILLING, activation='sequential'),
            PhaseSpec(QUARTERLY, activation='sequential', period=QUARTER),
        ]
        return System(
            self.label,
            elements=[premises, women],
            interactions=[day, *bills],
            space=LatticeSpace((self.width, self.height), boundary='fixed'),
            time=DiscreteTime(schedule=PhasesScheduleSpec(phases)),
        )
