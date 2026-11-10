"""Population and pollution.

The births, deaths, food, and quality of life of a population suffer from its pollution.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.laws.utils import series
from ....core.specs import DeclaredVariableSpec, ObservableSpec, ParameterSpec, StateSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations

ABSORPTION_TIME = ((0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0), (0.6, 2.5, 5.0, 8.0, 11.5, 15.5, 20.0))
BIRTHS_FROM_FOOD = ((0.0, 1.0, 2.0, 3.0, 4.0), (0.0, 1.0, 1.6, 1.9, 2.0))
BIRTHS_FROM_POLLUTION = ((0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0), (1.02, 0.9, 0.7, 0.4, 0.25, 0.15, 0.1))
DEATHS_FROM_FOOD = ((0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0), (30.0, 3.0, 2.0, 1.4, 1.0, 0.7, 0.6, 0.5, 0.5))
DEATHS_FROM_POLLUTION = ((0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0), (0.92, 1.3, 3.3, 4.8, 6.8, 9.2, 9.5))
FOOD_FROM_POLLUTION = ((0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0), (1.02, 0.9, 0.65, 0.35, 0.2, 0.1, 0.05))
CROPPING_INTENSITY = ((0.0, 10.0, 20.0, 30.0, 40.0, 50.0), (1.75, 1.8, 1.9, 2.0, 2.3, 2.5))
LIFE_FROM_FOOD = ((0.0, 1.0, 2.0, 3.0, 4.0), (0.0, 1.0, 1.8, 2.4, 2.7))
LIFE_FROM_POLLUTION = ((0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0), (1.05, 0.85, 0.6, 0.3, 0.15, 0.05, 0.02))


def _get_food_per_capita(s: SimpleNamespace) -> Expression:
    """Return the food per capita."""
    return _get_food(s, s.t)


def _get_quality_of_life(s: SimpleNamespace) -> Expression:
    """Return the quality of life.

    It is the product of the table multipliers of the food and the pollution ratios.
    """
    food_ratio = _get_food_per_capita(s) / s.subsistence_food
    ratio = _get_pollution_ratio(s)
    return series(food_ratio, *LIFE_FROM_FOOD) * series(ratio, *LIFE_FROM_POLLUTION)


def _get_food(region: SimpleNamespace, t: Expression) -> Expression:
    """Return the food per capita at a time.

    The food follows from the land, the yield, the cropping intensity, and the pollution.
    """
    ratio = _get_pollution_ratio(region)
    intensity = series(t, *CROPPING_INTENSITY)
    food = region.cultivated_land * region.crop_yield * intensity * series(ratio, *FOOD_FROM_POLLUTION)
    return food / region.population


def _get_pollution_ratio(region: SimpleNamespace) -> Expression:
    """Return the pollution ratio, the pollution over its standard."""
    return region.pollution / region.pollution_standard


class PopulationPollutionSystem(DynamicalSystem):
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
        >>> PopulationPollutionSystem(pollution_normal=0.5).prepare().system_.elements[0].get_value('pollution_normal')
        0.5
    """

    field = 'physics/climate'
    topics = ('pollution', 'population', 'stocks and flows', 'system dynamics', 'lookup tables', 'ode')
    references = ('Bala et al. 2017',)
    until = 50.0
    dt = 0.25
    observables = (
        ObservableSpec('food_per_capita', _get_food_per_capita, element='region'),
        ObservableSpec('quality_of_life', _get_quality_of_life, element='region'),
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
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.population = population
        self.pollution = pollution
        self.pollution_normal = pollution_normal
        self.pollution_standard = pollution_standard
        self.birth_normal = birth_normal
        self.death_normal = death_normal

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'region'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'population': DeclaredVariableSpec(self.population, StateSpec(domain=RealSet.closed_open(0, oo))),
            'pollution': DeclaredVariableSpec(self.pollution, StateSpec(domain=RealSet.closed_open(0, oo))),
            'pollution_normal': DeclaredVariableSpec(
                self.pollution_normal,
                ParameterSpec(domain=RealSet.closed_open(0, oo)),
            ),
            'pollution_standard': DeclaredVariableSpec(
                self.pollution_standard,
                ParameterSpec(domain=RealSet.open(0, oo)),
            ),
            'birth_normal': DeclaredVariableSpec(self.birth_normal, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'death_normal': DeclaredVariableSpec(self.death_normal, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'subsistence_food': DeclaredVariableSpec(180.0, ParameterSpec(domain=RealSet.open(0, oo))),
            'cultivated_land': DeclaredVariableSpec(7.4e6, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'crop_yield': DeclaredVariableSpec(2200.0, ParameterSpec(domain=RealSet.closed_open(0, oo))),
        }

    def _get_equations(self) -> Equations:
        """Return the flows of the population and the pollution."""
        return {'population': self._population_rate, 'pollution': self._pollution_rate}

    @staticmethod
    def _population_rate(s: SimpleNamespace) -> Expression:
        """Return the births minus the deaths of the population."""
        region = s.target
        ratio = _get_pollution_ratio(region)
        food_ratio = _get_food(region, s.t) / region.subsistence_food
        births = region.birth_normal * series(food_ratio, *BIRTHS_FROM_FOOD) * series(ratio, *BIRTHS_FROM_POLLUTION)
        deaths = region.death_normal * series(food_ratio, *DEATHS_FROM_FOOD) * series(ratio, *DEATHS_FROM_POLLUTION)
        return region.population * (births - deaths)

    @staticmethod
    def _pollution_rate(s: SimpleNamespace) -> Expression:
        """Return the generation minus the absorption of the pollution."""
        region = s.target
        absorption = region.pollution / series(_get_pollution_ratio(region), *ABSORPTION_TIME)
        return region.pollution_normal * region.population - absorption
