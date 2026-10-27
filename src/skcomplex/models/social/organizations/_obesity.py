"""Obesity models: people whose weight follows the meals they buy near home and work."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.laws import BaseLaw, RuleLaw
from ....core.spaces import ContinuousSpace
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.runtime import register_model
from ....engine.solvers import AgentView, ModelView

type Array = npt.NDArray[np.float64]

MEAL_RANGES = ((400.0, 900.0), (600.0, 1100.0), (800.0, 1500.0))
HABIT_RANGES = ((0.0, 0.5), (0.25, 0.75), (0.5, 1.0))
DAYS_PER_WEEK = 7
KCAL_PER_KG = 7700.0
OBESE = 30.0


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
        >>> from skcomplex.models.social.organizations import basal_metabolic_rate
        >>> round(basal_metabolic_rate(70.0, 175.0, 30.0, 1.0))
        1696
    """
    if male:
        return 88.362 + 13.397 * weight + 4.799 * height - 5.677 * age
    return 447.593 + 9.247 * weight + 3.098 * height - 4.330 * age


def _meal(kind: int, habit: int, rng: np.random.Generator) -> tuple[float, bool]:
    """Return the calories of a meal of a restaurant kind for an eating habit, and whether it is in the top half."""
    low, high = MEAL_RANGES[kind]
    start, stop = HABIT_RANGES[habit]
    calories = rng.uniform(low + start * (high - low), low + stop * (high - low))
    return float(calories), bool(calories > (low + high) / 2)


class RestaurantElement(Element):
    """Restaurant of the obesity model: a market at 0, a restaurant at 1, or a fast food outlet at 2.

    Args:
        label:
            The label of the agent or group.

        kind:
            The kind of restaurant.

        sold:
            The number of meals sold.

        sold_high:
            The number of meals sold from the top half of the restaurant's calorie range.

    Examples:
        >>> from skcomplex.models.social.organizations import RestaurantElement
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


class EaterElement(Element):
    """Person of the obesity model, who goes from home to work and back every day and eats near each place.

    At home, at work, and at home again the person eats one meal at a random restaurant within `reach`, or at the
    nearest restaurant when none is within reach, so that access changes what is eaten but never skips a meal. The
    calories are drawn from the bottom, middle, or top half of the restaurant's range by its eating habit. Every week
    its weight changes by the calories eaten beyond seven days of its expenditure, the basal metabolic rate times the
    `activity` factor of the Harris-Benedict equation, at 7700 kilocalories per kilogram.

    Args:
        label:
            The label of the agent or group.

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
            The number of days lived in the model.

        reach:
            The distance within which a person looks for a restaurant.

        activity:
            The factor of the basal metabolic rate a person spends a day, 1.2 when sedentary and 1.9 when very active.

    Examples:
        >>> from skcomplex.models.social.organizations import EaterElement
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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the daily rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.live)]

    def _eat(self, agent: AgentView, model: ModelView) -> None:
        """Eat one meal at a random restaurant within reach, or at the nearest restaurant if none is within reach."""
        restaurants = [other for other in agent.neighbors(radius=agent.reach) if other.group != agent.group]
        if not restaurants:
            nearest = min(model.agents('restaurants'), key=agent.distance_to, default=None)
            restaurants = [] if nearest is None else [nearest]
        if not restaurants:
            return
        restaurant = restaurants[int(model.rng.integers(len(restaurants)))]
        calories, high = _meal(int(restaurant.kind), int(agent.habit), model.rng)
        agent.calories += calories
        restaurant.sold += 1
        restaurant.sold_high += float(high)

    def _gain(self, agent: AgentView) -> None:
        """Change the weight by the calories of the week beyond the basal needs, and start a new week."""
        rate = basal_metabolic_rate(agent.weight, agent.height, agent.age, agent.male)
        needs = DAYS_PER_WEEK * agent.activity * rate
        agent.weight += (agent.calories - needs) / KCAL_PER_KG
        agent.calories = 0.0
        agent.bmi = agent.weight / (agent.height / 100) ** 2

    def live(self, agent: AgentView, model: ModelView) -> None:
        """Eat at home, at work, and at home again, and update the weight at the end of a week.

        Args:
            agent:
                The person.

            model:
                The model.
        """
        for place in (agent.home, agent.work, agent.home):
            agent.move_to(np.asarray(place, dtype=float))
            self._eat(agent, model)
        agent.day += 1
        if agent.day % DAYS_PER_WEEK == 0:
            self._gain(agent)


@register_model(
    field='social/organizations',
    tags=('health', 'agents', 'continuous space', 'obesity'),
    references=('Lynch et al. 2019', 'Carmichael et al. 2019'),
    defaults={'until': 70},
)
def obesity(
    n_people: int = 100,
    *,
    n_places: int = 20,
    size: float = 280.0,
    reach: float = 30.0,
    activity: float = 1.55,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the obesity model of adults who eat near home and work, one step per day.

    Homes, workplaces, and restaurants of three kinds lie at random, and the eating habits are split equally.

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

    Returns:
        system:
            The model, with the groups `people` and `restaurants`.

    Examples:
        >>> from skcomplex.models.social.organizations import obesity
        >>> [group.size for group in obesity(10, n_places=4, random_state=0).elements]
        [10, 4]
    """
    rng = np.random.default_rng(random_state)
    homes, works = rng.uniform(0, size, (2, n_places, 2))
    home, work = homes[rng.integers(n_places, size=n_people)], works[rng.integers(n_places, size=n_people)]
    male = (rng.random(n_people) < 0.5).astype(float)  # noqa: PLR2004
    height = np.where(male > 0, rng.normal(176, 7, n_people), rng.normal(163, 7, n_people))
    weight = rng.normal(24, 3, n_people) * (height / 100) ** 2
    people = GroupElement(
        'people',
        element=EaterElement,
        size=n_people,
        values={
            'home': home,
            'work': work,
            'weight': weight,
            'height': height,
            'age': rng.uniform(18, 65, n_people),
            'male': male,
            'habit': np.arange(n_people) % 3,
            'bmi': weight / (height / 100) ** 2,
            'reach': reach,
            'activity': activity,
        },
        location=home,
    )
    restaurants = GroupElement(
        'restaurants',
        element=RestaurantElement,
        size=n_places,
        values={'kind': rng.integers(0, 3, n_places).astype(float)},
        location=rng.uniform(0, size, (n_places, 2)),
    )
    return System(elements=[people, restaurants], space=ContinuousSpace(extent=(size, size)))
