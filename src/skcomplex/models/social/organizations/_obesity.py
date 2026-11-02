"""Obesity models: people whose weight follows the meals they buy near home and work."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.elements import GroupElement
from ....core.interactions import BaseInteraction, MovementInteraction, TransferInteraction
from ....core.spaces import ContinuousSpace, DiscreteTime
from ....core.system import System
from ....specs import PartnerSpec, PhaseSpec, PhasesScheduleSpec
from ..._catalog import register_model
from ._eater import DAYS_PER_WEEK, EaterElement
from ._restaurant import RestaurantElement

OBESE = 30.0


MALE_SHARE = 0.5


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

    Homes, workplaces, and restaurants of three kinds lie at random, and the eating habits are split equally. At home,
    at work, and at home again a person eats one meal at a random restaurant within `reach`, or at the nearest
    restaurant when none is within reach, so that access changes what is eaten but never skips a meal.

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
    male = (rng.random(n_people) < MALE_SHARE).astype(float)
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
        elements=[people, restaurants],
        interactions=interactions,
        space=ContinuousSpace(extent=(size, size)),
        time=DiscreteTime(schedule=PhasesScheduleSpec(phases)),
    )
