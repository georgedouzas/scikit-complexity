"""Person of the obesity system, who goes from home to work and back every day and eats near each place."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ._eater_view import EaterView
from ._restaurant_view import RestaurantView

MEAL_RANGES = ((400.0, 900.0), (600.0, 1100.0), (800.0, 1500.0))


HABIT_RANGES = ((0.0, 0.5), (0.25, 0.75), (0.5, 1.0))


DAYS_PER_WEEK = 7


KCAL_PER_KG = 7700.0


def basal_metabolic_rate(weight: float, height: float, age: float, male: float) -> float:
    """Return the daily basal metabolic rate of the revised Harris-Benedict equation, in kilocalories.

    Args:
        weight:
            The weight in kilograms.

        height:
            The height in centimetres.

        age:
            The age in years.

        male:
            1 for a man and 0 for a woman.

    Returns:
        rate:
            The basal metabolic rate per day.

    Examples:
        >>> from skcomplex.systems.social.organizations import basal_metabolic_rate
        >>> round(basal_metabolic_rate(70.0, 175.0, 30.0, 1.0))
        1696
    """
    if male:
        return 88.362 + 13.397 * weight + 4.799 * height - 5.677 * age
    return 447.593 + 9.247 * weight + 3.098 * height - 4.330 * age


class EaterElement(Element):
    """Person of the obesity system, who goes from home to work and back every day and eats near each place.

    The moves and the meals of the day are movements and transfers of the system, in the phases `home`, `work`, and
    `evening`. The calories of a meal are drawn from the bottom, middle, or top half of the restaurant's range by the
    eating habit. In the phase `week`, every seventh step, the weight changes by the calories eaten beyond seven days of
    the expenditure, the basal metabolic rate times the `activity` factor of the Harris-Benedict equation, at 7700
    kilocalories per kilogram.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        home:
            The position of the person's home.

        work:
            The position of the person's workplace.

        weight:
            The weight in kilograms.

        height:
            The height in centimetres.

        age:
            The age in years.

        male:
            1 for a man and 0 for a woman.

        habit:
            The eating habit: low at 0, medium at 1, or high at 2.

        calories:
            The calories eaten so far this week.

        bmi:
            The body mass index.

        day:
            The number of days lived in the system.

        reach:
            The distance within which a person looks for a restaurant.

        activity:
            The factor of the basal metabolic rate a person spends a day, 1.2 when sedentary and 1.9 when very active.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.organizations import EaterElement
        >>> group = GroupElement(
        ...     'people', element=EaterElement, size=5, values={'home': [1.0, 2.0], 'work': [3.0, 4.0]}
        ... )
        >>> group.label, group.element.__name__, group.size
        ('people', 'EaterElement', 5)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'bmi', 'color_range': (18.5, 35.0)}),
        *,
        home: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
        work: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
        weight: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 70.0,
        height: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 170.0,
        age: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 40.0,
        male: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 1.0,
        habit: Annotated[float, StateSpec(domain=NN)] = 1.0,
        calories: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        bmi: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 24.0,
        day: Annotated[float, StateSpec(domain=NN)] = 0.0,
        reach: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 30.0,
        activity: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.55,
    ) -> None:
        super().__init__(label, visual=visual)
        self.home = home
        self.work = work
        self.weight = weight
        self.height = height
        self.age = age
        self.male = male
        self.habit = habit
        self.calories = calories
        self.bmi = bmi
        self.day = day
        self.reach = reach
        self.activity = activity

    @classmethod
    def _get_view(cls) -> type[EaterView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return EaterView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule that counts the days and the rule that changes the weight once a week.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.count_day, phase='evening'), RuleLaw(self.gain_weight, phase='week')]

    def count_day(self, agent: EaterView, system: SystemView) -> None:
        """Count one more day lived in the system.

        Args:
            agent:
                The person.

            system:
                The system.
        """
        agent.day += 1

    def gain_weight(self, agent: EaterView, system: SystemView) -> None:
        """Change the weight by the calories of the week beyond the basal needs, and start a new week.

        Args:
            agent:
                The person.

            system:
                The system.
        """
        rate = basal_metabolic_rate(agent.weight, agent.height, agent.age, agent.male)
        needs = DAYS_PER_WEEK * agent.activity * rate
        agent.weight += (agent.calories - needs) / KCAL_PER_KG
        agent.calories = 0.0
        agent.bmi = agent.weight / (agent.height / 100) ** 2

    @staticmethod
    def go_home(agent: EaterView, system: SystemView) -> npt.NDArray[np.float64]:
        """Return the position of the home of a person.

        Args:
            agent:
                The person.

            system:
                The system.

        Returns:
            position:
                The position of the home.
        """
        return np.asarray(agent.home, dtype=float)

    @staticmethod
    def go_to_work(agent: EaterView, system: SystemView) -> npt.NDArray[np.float64]:
        """Return the position of the workplace of a person.

        Args:
            agent:
                The person.

            system:
                The system.

        Returns:
            position:
                The position of the workplace.
        """
        return np.asarray(agent.work, dtype=float)

    @staticmethod
    def draw_meal(agent: EaterView, restaurant: RestaurantView, system: SystemView) -> float:
        """Return the calories of a meal, drawn from the part of the range of a restaurant that a habit eats.

        Args:
            agent:
                The person.

            restaurant:
                The restaurant.

            system:
                The system.

        Returns:
            calories:
                The calories of the meal.
        """
        low, high = MEAL_RANGES[int(restaurant.kind)]
        start, stop = HABIT_RANGES[int(agent.habit)]
        return float(system.rng.uniform(low + start * (high - low), low + stop * (high - low)))

    @staticmethod
    def settle_meal(agent: EaterView, restaurant: RestaurantView, amounts: Sequence[float], system: SystemView) -> None:
        """Credit a person with the calories of a meal, and count the sale on the restaurant.

        Args:
            agent:
                The person.

            restaurant:
                The restaurant.

            amounts:
                The calories of the meal, the only amount.

            system:
                The system.
        """
        calories = amounts[0]
        low, high = MEAL_RANGES[int(restaurant.kind)]
        agent.calories += calories
        restaurant.sold += 1
        restaurant.sold_high += float(calories > (low + high) / 2)
