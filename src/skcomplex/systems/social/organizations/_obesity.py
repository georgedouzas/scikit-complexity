"""Obesity systems: people whose weight follows the meals they buy.

They buy the meals near home and work.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from functools import partial
from types import SimpleNamespace
from typing import Annotated, Protocol, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import BaseInteraction, Interaction
from ....core.components.laws import PartnerSpec, UpdateLaw
from ....core.components.laws.utils import pairs
from ....core.components.spaces import BaseSpace, ContinuousSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.compile import PhaseSpec
from ....core.engine.solvers import BaseSolver, MesaSolver, SystemView
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ...general.agents import MovementInteraction

MEAL_RANGES = ((400.0, 900.0), (600.0, 1100.0), (800.0, 1500.0))
HABIT_RANGES = ((0.0, 0.5), (0.25, 0.75), (0.5, 1.0))
DAYS_PER_WEEK = 7
KCAL_PER_KG = 7700.0
MALE_SHARE = 0.5


class EaterView(AgentViewProtocol, Protocol):
    """View of a person of the obesity system.

    Its states and parameters are attributes.
    """

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


class RestaurantView(AgentViewProtocol, Protocol):
    """View of a restaurant of the obesity system.

    Its states and parameters are attributes.
    """

    kind: float
    sold: float
    sold_high: float


def basal_metabolic_rate(weight: float, height: float, age: float, male: float) -> float:
    """Return the daily basal metabolic rate of the revised Harris-Benedict equation.

    It is in kilocalories.

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


type Meals = list[tuple[EaterView, RestaurantView]]


class EaterElement(Element):
    """Person of the obesity system.

    The person goes from home to work and back every day and eats near each place. The moves and the meals of the day
    are movements and transfers of the system, in the phases `home`, `work`, and `evening`. The calories of a meal are
    drawn from the bottom, middle, or top half of the restaurant's range by the eating habit. In the phase `week`, every
    seventh step, the weight changes by the calories eaten beyond seven days of the expenditure, the basal metabolic
    rate times the `activity` factor of the Harris-Benedict equation, at 7700 kilocalories per kilogram.

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
            The number of days lived in the system.

        reach:
            The distance within which a person looks for a restaurant.

        activity:
            The factor of the basal metabolic rate a person spends a day, 1.2 when sedentary and 1.9 when very active.

        values:
            Further values by name, such as the position `x` and `y` of every member.

        size:
            The number of members, or `None` for one.

        space:
            The plane whose axes `x` and `y` place the members, or `None` for none.

    Examples:
        >>> from skcomplex.systems.social.organizations import EaterElement
        >>> group = EaterElement('people', home=[1.0, 2.0], work=[3.0, 4.0], size=5)
        >>> group.label, group.size
        ('people', 5)
    """

    visual = VisualSpec('agents', {'color_by': 'bmi', 'color_range': (18.5, 35.0)})

    def __init__(
        self,
        label: str,
        *,
        home: npt.ArrayLike = 0.0,
        work: npt.ArrayLike = 0.0,
        weight: Annotated[object, RealSet.open(0, oo)] = 70.0,
        height: Annotated[object, RealSet.open(0, oo)] = 170.0,
        age: Annotated[object, RealSet.open(0, oo)] = 40.0,
        male: Annotated[object, FiniteEnumeratedSet([0, 1])] = 1.0,
        habit: Annotated[object, NN] = 1.0,
        calories: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        bmi: Annotated[object, RealSet.open(0, oo)] = 24.0,
        day: Annotated[object, NN] = 0.0,
        reach: Annotated[object, RealSet.open(0, oo)] = 30.0,
        activity: Annotated[object, RealSet.open(0, oo)] = 1.55,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
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
        """Return the calories of a meal.

        They are drawn from the part of the range of a restaurant that a habit eats.

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
        """Credit a person with the calories of a meal.

        The sale is counted on the restaurant.

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


class DietInteraction(Interaction):
    """Days and weekly weight change of the people of the obesity system.

    Every evening a person counts one more day lived in the system, and once a week its weight changes by the calories
    of the week beyond its basal needs, and a new week starts.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of people.

        day_phase:
            The phase the days are counted in.

        week_phase:
            The phase the weight changes in.

    Examples:
        >>> from skcomplex.systems.social.organizations import DietInteraction
        >>> DietInteraction('diet', target='people').week_phase
        'week'
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        day_phase: str | None = 'evening',
        week_phase: str | None = 'week',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.day_phase = day_phase
        self.week_phase = week_phase

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the counting of the days and the change of the weight as the rules of every person."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [
                UpdateLaw(context.target, self.count_day, phase=self.day_phase),
                UpdateLaw(context.target, self.gain_weight, phase=self.week_phase),
            ]

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
        """Change the weight by the calories of the week, and start a new week.

        Only the calories beyond the basal needs count.

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


class RestaurantElement(Element):
    """Restaurant of the obesity system.

    It is a market at 0, a restaurant at 1, or a fast food outlet at 2.

    Args:
        label:
            The label of the agent or group.

        kind:
            The kind of restaurant.

        sold:
            The number of meals sold.

        sold_high:
            The number of meals sold from the top half of the restaurant's calorie range.

        values:
            Further values by name, such as the position `x` and `y` of every member.

        size:
            The number of members, or `None` for one.

        space:
            The plane whose axes `x` and `y` place the members, or `None` for none.

    Examples:
        >>> from skcomplex.systems.social.organizations import RestaurantElement
        >>> group = RestaurantElement('restaurants', kind=2.0, size=3)
        >>> group.label, group.size
        ('restaurants', 3)
    """

    def __init__(
        self,
        label: str,
        *,
        kind: Annotated[object, NN] = 0.0,
        sold: Annotated[object, NN] = 0.0,
        sold_high: Annotated[object, NN] = 0.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.kind = kind
        self.sold = sold
        self.sold_high = sold_high


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_people = n_people
        self.n_places = n_places
        self.size = size
        self.reach = reach
        self.activity = activity
        self.random_state = random_state

    @staticmethod
    def _eat(system: SystemView) -> None:
        """Let every person eat a meal at a restaurant within reach, or the nearest one, in random order."""
        people = cast(list[EaterView], system.get_agents('people'))
        restaurants = cast(list[RestaurantView], system.get_agents('restaurants'))
        found = cast(Meals, list(pairs(people, restaurants, system.rng, PartnerSpec('reach', fallback='nearest'))))
        for index in system.rng.permutation(len(found)):
            eater, restaurant = found[int(index)]
            if eater.alive and restaurant.alive:
                calories = EaterElement.draw_meal(eater, restaurant, system)
                EaterElement.settle_meal(eater, restaurant, [calories], system)

    @staticmethod
    def _get_meals(s: SimpleNamespace, phase: str) -> list[BaseComponent]:
        """Return the meals of a phase as the rule of the system."""
        return [UpdateLaw(s, ObesitySystem._eat, phase=phase)]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        homes, works = rng.uniform(0, self.size, (2, self.n_places, 2))
        home, work = (
            homes[rng.integers(self.n_places, size=self.n_people)],
            works[rng.integers(self.n_places, size=self.n_people)],
        )
        male = (rng.random(self.n_people) < MALE_SHARE).astype(float)
        height = np.where(male > 0, rng.normal(176, 7, self.n_people), rng.normal(163, 7, self.n_people))
        weight = rng.normal(24, 3, self.n_people) * (height / 100) ** 2
        space = ContinuousSpace(extent=(self.size, self.size))
        people = EaterElement(
            'people',
            home=home,
            work=work,
            weight=weight,
            height=height,
            age=rng.uniform(18, 65, self.n_people),
            male=male,
            habit=np.arange(self.n_people) % 3,
            bmi=weight / (height / 100) ** 2,
            reach=self.reach,
            activity=self.activity,
            values={'x': home[:, 0], 'y': home[:, 1]},
            size=self.n_people,
            space=space,
        )
        kinds = rng.integers(0, 3, self.n_places).astype(float)
        x, y = rng.uniform(0, self.size, (self.n_places, 2)).T
        restaurants = RestaurantElement(
            'restaurants',
            kind=kinds,
            values={'x': x, 'y': y},
            size=self.n_places,
            space=space,
        )
        interactions: list[BaseInteraction] = [DietInteraction('diet', target='people')]
        for phase, step in (
            ('home', EaterElement.go_home),
            ('work', EaterElement.go_to_work),
            ('evening', EaterElement.go_home),
        ):
            interactions += [
                MovementInteraction(f'go_{phase}', target='people', step=step, phase=phase),
                Interaction(
                    f'meal_{phase}',
                    roles={'people': 'people', 'restaurants': 'restaurants'},
                    laws=partial(self._get_meals, phase=phase),
                ),
            ]
        phases = (PhaseSpec('home'), PhaseSpec('work'), PhaseSpec('evening'), PhaseSpec('week', period=DAYS_PER_WEEK))
        self.system_ = System(
            self.label,
            elements=[people, restaurants],
            interactions=interactions,
            space=space,
            solver=MesaSolver(phases=phases),
        )
