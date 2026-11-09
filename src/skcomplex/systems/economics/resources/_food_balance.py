"""Resources systems: rice land cultivated towards the land available.

It is set against the food a growing population needs.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import TransferInteraction
from ....core.components.laws import Sink, Source
from ....core.components.laws.utils import where
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec


def _food_available(s: SimpleNamespace) -> Expression:
    """Return the rice available, the area cultivated times the yield."""
    return s.area_cultivated * s.crop_yield


def _food_required(s: SimpleNamespace) -> Expression:
    """Return the rice required, the population times the rice per capita."""
    return s.population * s.food_per_capita


def _food_security(s: SimpleNamespace) -> Expression:
    """Return the food security level, the rice available over the rice required."""
    return _food_available(s) / _food_required(s)


class RiceBalanceElement(Element):
    """Rice land and population of a region.

    It holds the land available for rice and the rice each person needs. The area cultivated grows towards the land
    available over a planning time and loses a fraction to housing, roads, and industry, and the population grows by
    births and falls by deaths, through transfers of the system.

    Args:
        label:
            The label of the region.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        area_cultivated:
            The area under rice, in hectares.

        population:
            The population.

        total_land_available:
            The land suitable for rice, in hectares.

        planning_time:
            The years over which the land suitable for rice is to be covered.

        land_fraction_lost:
            The fraction of the area cultivated lost every year.

        crop_yield:
            The yield of rice, in tonnes per hectare.

        birth_fraction:
            The births per person and year.

        life_expectancy:
            The life expectancy, in years.

        food_per_capita:
            The rice each person needs, in tonnes per year.

    Examples:
        >>> from skcomplex.systems.economics.resources import RiceBalanceElement
        >>> RiceBalanceElement('region', crop_yield=4.0).crop_yield
        4.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        area_cultivated: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo), quantity='land')] = 68000.0,
        population: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo), quantity='people')] = 2800000.0,
        total_land_available: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 120000.0,
        planning_time: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 15.0,
        land_fraction_lost: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00085,
        crop_yield: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.2,
        birth_fraction: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.025,
        life_expectancy: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 65.0,
        food_per_capita: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.18,
    ) -> None:
        super().__init__(label, visual=visual)
        self.area_cultivated = area_cultivated
        self.population = population
        self.total_land_available = total_land_available
        self.planning_time = planning_time
        self.land_fraction_lost = land_fraction_lost
        self.crop_yield = crop_yield
        self.birth_fraction = birth_fraction
        self.life_expectancy = life_expectancy
        self.food_per_capita = food_per_capita


class RiceFoodBalanceSystem(BaseSystem):
    """Bala's food security model.

    Rice land grows towards the land available against a growing population. The area cultivated grows at
    `MAX((total_land_available - area_cultivated) / planning_time, 0)` and loses `land_fraction_lost` of itself every
    year to housing, roads, and industry. The population grows by `birth_fraction` per person and falls by one over the
    life expectancy. The food security level is the rice available, the area times the yield, over the rice required,
    the population times the rice per capita, and below 1 the region is short of rice. The structure, the equations, and
    the defaults are those of the STELLA model of Bala et al. (2017, Section 4.11.3), in hectares, people, tonnes, and
    years. The area rises for about fifty years towards the land available, and the food security rises for about
    twenty-five years and then falls as the population grows.

    The system has the stocks `region.area_cultivated` and `region.population`, and the flows
    `region.area_increase_rate`, `region.land_lost`, `region.birth_rate`, and `region.death_rate` per year. The
    observables of the system are `region.food_available`, `region.food_required`, and `region.food_security`.

    Args:
        area_cultivated:
            The initial area under rice, in hectares.

        population:
            The initial population.

        total_land_available:
            The land suitable for rice, in hectares.

        planning_time:
            The years over which the land suitable for rice is to be covered.

        land_fraction_lost:
            The fraction of the area cultivated lost every year.

        crop_yield:
            The yield of rice, in tonnes per hectare.

        birth_fraction:
            The births per person and year.

        life_expectancy:
            The life expectancy, in years.

        food_per_capita:
            The rice each person needs, in tonnes per year.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.resources import RiceFoodBalanceSystem
        >>> RiceFoodBalanceSystem(planning_time=10.0).prepare().system_.elements[0].planning_time
        10.0
    """

    field = 'economics/resources'
    topics = ('food security', 'agriculture', 'land use', 'stocks and flows', 'system dynamics', 'ode')
    references = ('Bala et al. 2017',)
    until = 100.0
    dt = 0.25
    observables = tuple(
        [
            ObservableSpec(name, expression, element='region')
            for name, expression in (
                ('food_available', _food_available),
                ('food_required', _food_required),
                ('food_security', _food_security),
            )
        ],
    )

    def __init__(
        self,
        area_cultivated: float = 68000.0,
        population: float = 2800000.0,
        *,
        total_land_available: float = 120000.0,
        planning_time: float = 15.0,
        land_fraction_lost: float = 0.00085,
        crop_yield: float = 3.2,
        birth_fraction: float = 0.025,
        life_expectancy: float = 65.0,
        food_per_capita: float = 0.18,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.area_cultivated = area_cultivated
        self.population = population
        self.total_land_available = total_land_available
        self.planning_time = planning_time
        self.land_fraction_lost = land_fraction_lost
        self.crop_yield = crop_yield
        self.birth_fraction = birth_fraction
        self.life_expectancy = life_expectancy
        self.food_per_capita = food_per_capita

    @staticmethod
    def _area_increase(s: SimpleNamespace) -> Expression:
        """Return the area increase rate.

        It is `MAX((total_land_available - area_cultivated) / planning_time, 0)`.
        """
        gap = (s.source.total_land_available - s.source.area_cultivated) / s.source.planning_time
        return where(gap > 0, gap, 0)

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        region = RiceBalanceElement(
            'region',
            area_cultivated=self.area_cultivated,
            population=self.population,
            total_land_available=self.total_land_available,
            planning_time=self.planning_time,
            land_fraction_lost=self.land_fraction_lost,
            crop_yield=self.crop_yield,
            birth_fraction=self.birth_fraction,
            life_expectancy=self.life_expectancy,
            food_per_capita=self.food_per_capita,
        )
        land = TransferInteraction(
            'land',
            source='region',
            target='region',
            state=Source('suitable_land'),
            into='area_cultivated',
            amount=self._area_increase,
            flow='area_increase_rate',
            legs=[
                (
                    'area_cultivated',
                    Sink('urban_land'),
                    lambda s: s.source.area_cultivated * s.source.land_fraction_lost,
                    'land_lost',
                ),
            ],
        )
        people = TransferInteraction(
            'people',
            source='region',
            target='region',
            state=Source('births'),
            into='population',
            amount=lambda s: s.source.population * s.source.birth_fraction,
            flow='birth_rate',
            legs=[
                ('population', Sink('deaths'), lambda s: s.source.population / s.source.life_expectancy, 'death_rate'),
            ],
        )
        return CompositeSystem(self.label, elements=[region], interactions=[land, people])
