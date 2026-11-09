"""Population and pollution.

The births, deaths, food, and quality of life of a population suffer from its pollution.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import lookup
from ....core.components.systems import BaseSystem, System
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec

ABSORPTION_TIME = ((0.0, 0.6), (10.0, 2.5), (20.0, 5.0), (30.0, 8.0), (40.0, 11.5), (50.0, 15.5), (60.0, 20.0))
BIRTHS_FROM_FOOD = ((0.0, 0.0), (1.0, 1.0), (2.0, 1.6), (3.0, 1.9), (4.0, 2.0))
BIRTHS_FROM_POLLUTION = ((0.0, 1.02), (10.0, 0.9), (20.0, 0.7), (30.0, 0.4), (40.0, 0.25), (50.0, 0.15), (60.0, 0.1))
DEATHS_FROM_FOOD = (
    (0.0, 30.0),
    (0.25, 3.0),
    (0.5, 2.0),
    (0.75, 1.4),
    (1.0, 1.0),
    (1.25, 0.7),
    (1.5, 0.6),
    (1.75, 0.5),
    (2.0, 0.5),
)
DEATHS_FROM_POLLUTION = ((0.0, 0.92), (10.0, 1.3), (20.0, 3.3), (30.0, 4.8), (40.0, 6.8), (50.0, 9.2), (60.0, 9.5))
FOOD_FROM_POLLUTION = ((0.0, 1.02), (10.0, 0.9), (20.0, 0.65), (30.0, 0.35), (40.0, 0.2), (50.0, 0.1), (60.0, 0.05))
CROPPING_INTENSITY = ((0.0, 1.75), (10.0, 1.8), (20.0, 1.9), (30.0, 2.0), (40.0, 2.3), (50.0, 2.5))
LIFE_FROM_FOOD = ((0.0, 0.0), (1.0, 1.0), (2.0, 1.8), (3.0, 2.4), (4.0, 2.7))
LIFE_FROM_POLLUTION = ((0.0, 1.05), (10.0, 0.85), (20.0, 0.6), (30.0, 0.3), (40.0, 0.15), (50.0, 0.05), (60.0, 0.02))


class PollutedRegionElement(Element):
    """Population of a region and the pollution it generates.

    The environment absorbs the pollution ever more slowly.

    The pollution grows by `pollution_normal` per person and is absorbed over a time that a table lengthens with the
    pollution ratio, the pollution over its standard. The births and the deaths are the normal rates times table
    multipliers of the food ratio, the food per capita over the subsistence food, and of the pollution ratio. The food
    is the cultivated land times the yield, a cropping intensity that a table raises over time, and a table
    multiplier of the pollution ratio.

    Args:
        label:
            The label of the region.

        visual:
            How the region is drawn, or `None` for `default_visual_`.

        population:
            The initial population.

        pollution:
            The initial pollution.

        pollution_normal:
            The pollution one person generates per year.

        pollution_standard:
            The pollution that the pollution ratio divides by.

        birth_normal:
            The normal birth rate per year.

        death_normal:
            The normal death rate per year.

        subsistence_food:
            The food per capita that the food ratio divides by.

        cultivated_land:
            The cultivated land.

        crop_yield:
            The yield of the land per crop.

    Attributes:
        default_visual_:
            The visual of the region when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.climate import PollutedRegionElement
        >>> PollutedRegionElement('region').birth_normal
        0.04
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        population: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1e8,
        pollution: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1e7,
        pollution_normal: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.25,
        pollution_standard: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.5e7,
        birth_normal: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.04,
        death_normal: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.028,
        subsistence_food: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 180.0,
        cultivated_land: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 7.4e6,
        crop_yield: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2200.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.population = population
        self.pollution = pollution
        self.pollution_normal = pollution_normal
        self.pollution_standard = pollution_standard
        self.birth_normal = birth_normal
        self.death_normal = death_normal
        self.subsistence_food = subsistence_food
        self.cultivated_land = cultivated_land
        self.crop_yield = crop_yield

    @staticmethod
    def _pollution_ratio(s: SimpleNamespace) -> Expression:
        """Return the pollution ratio, the pollution over its standard."""
        return s.pollution / s.pollution_standard

    @staticmethod
    def _food_per_capita(s: SimpleNamespace) -> Expression:
        """Return the food per capita.

        The food follows from the land, the yield, the cropping intensity, and the pollution.
        """
        ratio = PollutedRegionElement._pollution_ratio(s)
        intensity = lookup(s.t, CROPPING_INTENSITY)
        food = s.cultivated_land * s.crop_yield * intensity * lookup(ratio, FOOD_FROM_POLLUTION)
        return food / s.population

    @staticmethod
    def _quality_of_life(s: SimpleNamespace) -> Expression:
        """Return the quality of life.

        It is the product of the table multipliers of the food and the pollution ratios.
        """
        food_ratio = PollutedRegionElement._food_per_capita(s) / s.subsistence_food
        ratio = PollutedRegionElement._pollution_ratio(s)
        return lookup(food_ratio, LIFE_FROM_FOOD) * lookup(ratio, LIFE_FROM_POLLUTION)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the flows of the population and the pollution.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of the two stocks.
        """
        ratio = self._pollution_ratio(s)
        food_ratio = self._food_per_capita(s) / s.subsistence_food
        births = s.birth_normal * lookup(food_ratio, BIRTHS_FROM_FOOD) * lookup(ratio, BIRTHS_FROM_POLLUTION)
        deaths = s.death_normal * lookup(food_ratio, DEATHS_FROM_FOOD) * lookup(ratio, DEATHS_FROM_POLLUTION)
        absorption = s.pollution / lookup(ratio, ABSORPTION_TIME)
        return [
            RateLaw(s.population, s.population * (births - deaths)),
            RateLaw(s.pollution, s.pollution_normal * s.population - absorption),
        ]


class PopulationPollutionSystem(BaseSystem):
    """The pollution model of Bala, Arshad, and Noh.

    A population degrades its environment.

    The population generates pollution at `pollution_normal` per person per year, and the environment absorbs the
    pollution over a time that lengthens from 0.6 to 20 years as the pollution ratio, pollution over its standard, rises
    from 0 to 60. This makes the absorption saturate, so a growing population can drive the pollution up steeply.
    Pollution lowers births, food, and the quality of life, and raises deaths, through table multipliers, while food per
    capita over the subsistence food raises births and lowers deaths. The cropping intensity rises from 1.75 to 2.5
    crops per year over 50 years. The model follows Section 4.11.9 of Bala et al. (2017), with its tables and
    parameters, and time in years. The system has the stocks `region.population` and `region.pollution`, and the
    observables `region.food_per_capita` and `region.quality_of_life`.

    Args:
        population:
            The initial population.

        pollution:
            The initial pollution.

        pollution_normal:
            The pollution one person generates per year.

        pollution_standard:
            The pollution that the pollution ratio divides by.

        birth_normal:
            The normal birth rate per year.

        death_normal:
            The normal death rate per year.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.climate import PopulationPollutionSystem
        >>> PopulationPollutionSystem(pollution_normal=0.5).prepare().system_.elements[0].pollution_normal
        0.5
    """

    field = "physics/climate"
    topics = ("pollution", "population", "stocks and flows", "system dynamics", "lookup tables", "ode")
    references = ("Bala et al. 2017",)
    until = 50.0
    dt = 0.25
    observables = (
        ObservableSpec("food_per_capita", PollutedRegionElement._food_per_capita, element="region"),
        ObservableSpec("quality_of_life", PollutedRegionElement._quality_of_life, element="region"),
    )

    def __init__(
        self,
        population: float = 1e8,
        pollution: float = 1e7,
        *,
        pollution_normal: float = 0.25,
        pollution_standard: float = 1.5e7,
        birth_normal: float = 0.04,
        death_normal: float = 0.028,
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.population = population
        self.pollution = pollution
        self.pollution_normal = pollution_normal
        self.pollution_standard = pollution_standard
        self.birth_normal = birth_normal
        self.death_normal = death_normal

    def _build_system(self) -> System:
        """Return the pollution model of Bala, Arshad, and Noh.

        A population degrades its environment.
        """
        region = PollutedRegionElement(
            "region",
            population=self.population,
            pollution=self.pollution,
            pollution_normal=self.pollution_normal,
            pollution_standard=self.pollution_standard,
            birth_normal=self.birth_normal,
            death_normal=self.death_normal,
        )
        return System(self.label, elements=[region])
