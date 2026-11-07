"""Obesity systems: people whose weight follows the meals they buy near home and work."""

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

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import BaseInteraction, MovementInteraction, TransferInteraction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.spaces import ContinuousSpace, DiscreteTime
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ParameterSpec, PartnerSpec, PhaseSpec, PhasesScheduleSpec, StateSpec, VisualSpec
from ....core.types import Array


class EaterView(AgentView):
    """View of a person of the obesity system, with its states and parameters as attributes."""

    home: Array
    work: Array
    weight: float
    height: float
    age: float
    male: float
    habit: float
    calories: float
    bmi: float
    day: float
    reach: float
    activity: float


class RestaurantView(AgentView):
    """View of a restaurant of the obesity system, with its states and parameters as attributes."""

    kind: float
    sold: float
    sold_high: float


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

    law_classes = (RuleLaw,)

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


class RestaurantElement(Element):
    """Restaurant of the obesity system: a market at 0, a restaurant at 1, or a fast food outlet at 2.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        kind:
            The kind of restaurant.

        sold:
            The number of meals sold.

        sold_high:
            The number of meals sold from the top half of the restaurant's calorie range.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.organizations import RestaurantElement
        >>> group = GroupElement('restaurants', element=RestaurantElement, size=3, values={'kind': 2.0})
        >>> group.label, group.element.__name__, group.size
        ('restaurants', 'RestaurantElement', 3)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        kind: Annotated[float, StateSpec(domain=NN)] = 0.0,
        sold: Annotated[float, StateSpec(domain=NN)] = 0.0,
        sold_high: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.kind = kind
        self.sold = sold
        self.sold_high = sold_high

    @classmethod
    def _get_view(cls) -> type[RestaurantView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return RestaurantView


OBESE = 30.0


MALE_SHARE = 0.5


class ObesitySystem(BaseSystem):
    """The obesity system of adults who eat near home and work, one step per day.

    Homes, workplaces, and restaurants of three kinds lie at random, and the eating habits are split equally. At home,
    at work, and at home again a person eats one meal at a random restaurant within `reach`, or at the nearest
    restaurant when none is within reach, so that access changes what is eaten but never skips a meal. The system has
    the groups `people` and `restaurants`.

    Args:
        n_people:
            The number of people.

        n_places:
            The number of homes, of workplaces, and of restaurants.

        size:
            The side of the square the places lie in.

        reach:
            The distance within which a person looks for a restaurant.

        activity:
            The factor of the basal metabolic rate a person spends a day, 1.55 for moderate activity.

        random_state:
            The seed or generator of the places and the people.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.organizations import ObesitySystem
        >>> [group.size for group in ObesitySystem(10, n_places=4, random_state=0).prepare().system_.elements]
        [10, 4]
    """

    field = 'social/organizations'
    topics = ('health', 'agents', 'continuous space', 'obesity')
    references = ('Lynch et al. 2019', 'Carmichael et al. 2019')
    until = 70

    def __init__(
        self,
        n_people: int = 100,
        *,
        n_places: int = 20,
        size: float = 280.0,
        reach: float = 30.0,
        activity: float = 1.55,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_people = n_people
        self.n_places = n_places
        self.size = size
        self.reach = reach
        self.activity = activity
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        homes, works = rng.uniform(0, self.size, (2, self.n_places, 2))
        home, work = (
            homes[rng.integers(self.n_places, size=self.n_people)],
            works[rng.integers(self.n_places, size=self.n_people)],
        )
        male = (rng.random(self.n_people) < MALE_SHARE).astype(float)
        height = np.where(male > 0, rng.normal(176, 7, self.n_people), rng.normal(163, 7, self.n_people))
        weight = rng.normal(24, 3, self.n_people) * (height / 100) ** 2
        people = GroupElement(
            'people',
            element=EaterElement,
            size=self.n_people,
            values={
                'home': home,
                'work': work,
                'weight': weight,
                'height': height,
                'age': rng.uniform(18, 65, self.n_people),
                'male': male,
                'habit': np.arange(self.n_people) % 3,
                'bmi': weight / (height / 100) ** 2,
                'reach': self.reach,
                'activity': self.activity,
            },
            location=home,
        )
        restaurants = GroupElement(
            'restaurants',
            element=RestaurantElement,
            size=self.n_places,
            values={'kind': rng.integers(0, 3, self.n_places).astype(float)},
            location=rng.uniform(0, self.size, (self.n_places, 2)),
        )
        interactions: list[BaseInteraction] = []
        for phase, step in (
            ('home', EaterElement.go_home),
            ('work', EaterElement.go_to_work),
            ('evening', EaterElement.go_home),
        ):
            interactions += [
                MovementInteraction(f'go_{phase}', target='people', step=step, phase=phase),
                TransferInteraction(
                    f'meal_{phase}',
                    source='people',
                    target='restaurants',
                    state='calories',
                    into='sold',
                    amount=EaterElement.draw_meal,
                    pairing=PartnerSpec('reach', fallback='nearest'),
                    form='agent',
                    phase=phase,
                    settlement=EaterElement.settle_meal,
                ),
            ]
        phases = (PhaseSpec('home'), PhaseSpec('work'), PhaseSpec('evening'), PhaseSpec('week', period=DAYS_PER_WEEK))
        return System(
            self.label,
            elements=[people, restaurants],
            interactions=interactions,
            space=ContinuousSpace(extent=(self.size, self.size)),
            time=DiscreteTime(schedule=PhasesScheduleSpec(phases)),
        )
