"""Driver agents: cars whose drivers choose their speed from the distance to the car ahead."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar, cast

import numpy as np

from ....core.base import Domain, VisualSpec, parameter, role, state
from ....core.model import AgentType, ContinuousSpace, Interaction, Law, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

ROAD_WIDTH = 1.0


def _position(agent: AgentView) -> tuple[float, float]:
    """Return the position of an agent in a continuous space."""
    x, y = np.asarray(agent.location, dtype=float)
    return float(x), float(y)


def _car_ahead(agent: AgentView, model: ModelView, n_cars: int) -> AgentView | None:
    """Return the car whose rank follows the agent's rank on a road of `n_cars` cars, or `None` for the last car."""
    rank = (int(agent.rank) + 1) % n_cars if n_cars else 0
    return next((car for car in model.agents(agent.group) if int(car.rank) == rank and car != agent), None)


class Motorist(AgentType):
    """Car of Charbonneau's traffic model, which brakes when it is closer than a safe distance to the car ahead.

    Args:
        label:
            The label of the agent type or group.

        x:
            The position along the road.

        v:
            The speed.

        rank:
            The order of the car along the road, 0 for the last car.

        safe_distance:
            The distance to the car ahead below which the driver brakes.

        braking:
            The speed lost when braking.

        acceleration:
            The speed gained when accelerating.

        speed_limit:
            The largest speed.

    Examples:
        >>> from skcomplex.models.engineering.traffic import Motorist
        >>> Motorist.group('cars', 3, x=[0.0, 10.0, 20.0]).size
        3
    """

    x: float = state(default=0.0)
    v: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    rank: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    safe_distance: float = parameter(default=5.0, domain=Domain.POSITIVE)
    braking: float = parameter(default=3.0, domain=Domain.NONNEGATIVE)
    acceleration: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    speed_limit: float = parameter(default=10.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the speed adjustment rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.adjust)]

    def adjust(self, agent: AgentView, model: ModelView) -> None:
        """Brake when the car ahead is closer than the safe distance and accelerate otherwise.

        The lead car, which has no car ahead, keeps its distance to the car behind it instead.

        Args:
            agent:
                The car.

            model:
                The model.
        """
        cars = model.agents(agent.group)
        ahead = next((car for car in cars if int(car.rank) == int(agent.rank) + 1), None)
        if ahead is None:
            behind = next((car for car in cars if int(car.rank) == int(agent.rank) - 1), None)
            distance = agent.x - behind.x if behind is not None else np.inf
        else:
            distance = ahead.x - agent.x
        change = -agent.braking if distance < agent.safe_distance else agent.acceleration
        agent.v = float(np.clip(agent.v + change, 0.0, agent.speed_limit))


class CarelessBraking(Interaction):
    """Model rule of Charbonneau's traffic model: a random driver brakes now and then, and every car moves.

    Every step, with probability `probability`, one random driver brakes by the braking of its group whatever the
    distance to the car ahead. Then every car moves by its speed, but no closer than `gap` to the car ahead.

    Args:
        label:
            The label of the interaction.

        cars:
            The path of the group of cars.

        probability:
            The probability that a random driver brakes in a step.

        gap:
            The smallest distance a car keeps to the car ahead.

    Examples:
        >>> from skcomplex.models.engineering.traffic import CarelessBraking
        >>> CarelessBraking('bozo', cars='cars').probability
        0.1
    """

    cars: str = role()
    probability: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    gap: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rule of random braking and motion.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.drive)]

    def drive(self, model: ModelView) -> None:
        """Make a random driver brake with the interaction's probability, then move every car.

        Args:
            model:
                The model.
        """
        cars = sorted(model.agents('cars'), key=lambda car: car.rank)
        if not cars:
            return
        if model.rng.random() < model.parameter('probability'):
            careless = cars[int(model.rng.integers(len(cars)))]
            careless.v = max(careless.v - careless.braking, 0.0)
        gap = model.parameter('gap')
        positions = [car.x for car in cars]
        for index, car in enumerate(cars):
            moved = car.x + car.v
            car.x = min(moved, positions[index + 1] - gap) if index + 1 < len(cars) else moved


class Driver(AgentType):
    """Driver on a ring road who accelerates as much as allowed and stops when the speed would cause a collision.

    The speed changes by the acceleration, bounded by `min_acc` and `max_acc`, is multiplied by a random factor between
    `1 - eps` and `1 + eps`, and stays between 0 and the speed limit. A speed larger than the distance to the car ahead
    becomes 0. The car then moves along the ring by its speed, and the ring is as long as the width of the space.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial position on the ring road, as a point `(x, y)` of the road strip.

        speed:
            The speed.

        rank:
            The order of the car along the ring.

        acceleration:
            The acceleration every driver chooses.

        min_acc:
            The largest braking, as a negative acceleration.

        max_acc:
            The largest acceleration.

        speed_limit:
            The largest speed.

        eps:
            The relative amplitude of the random error of speed.

    Examples:
        >>> from skcomplex.models.engineering.traffic import Driver
        >>> Driver.group('drivers', 2, location=[(0.0, 0.5), (10.0, 0.5)]).size
        2
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'speed'})
    speed: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    rank: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    acceleration: float = parameter(default=1.0)
    min_acc: float = parameter(default=-10.0)
    max_acc: float = parameter(default=1.0)
    speed_limit: float = parameter(default=40.0, domain=Domain.POSITIVE)
    eps: float = parameter(default=0.02, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the driving rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.drive)]

    def drive(self, agent: AgentView, model: ModelView) -> None:
        """Choose the speed from the distance to the car ahead and move along the ring.

        Args:
            agent:
                The driver.

            model:
                The model.
        """
        cars = model.agents(agent.group)
        length = cast('tuple[float, float]', model.dimensions)[0]
        ahead = _car_ahead(agent, model, len(cars))
        x, y = _position(agent)
        distance = (_position(ahead)[0] - x) % length if ahead is not None else length
        acceleration = min(max(agent.acceleration, agent.min_acc), agent.max_acc)
        speed = (agent.speed + acceleration) * model.rng.uniform(1 - agent.eps, 1 + agent.eps)
        speed = min(max(speed, 0.0), agent.speed_limit)
        agent.speed = 0.0 if speed > distance else speed
        agent.move_to(((x + agent.speed) % length, y))


@register_model(
    field='engineering/traffic',
    tags=('traffic', 'jams', 'agents', 'avalanches', 'stochastic'),
    references=('Charbonneau 2017',),
    defaults={'until': 100},
)
def charbonneau_traffic(
    n_cars: int = 100,
    *,
    probability: float = 0.1,
    safe_distance: float = 5.0,
    speed_limit: float = 10.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Charbonneau's traffic model, where random braking of single drivers sets off traffic jams.

    Cars on a one-way road start at rest with random spacings between 3 and 17. Every step, each driver brakes by 3
    when the car ahead is closer than the safe distance and accelerates by 1 otherwise, within the speed limit. A random
    driver brakes with a probability every step, and then every car moves by its speed.

    Args:
        n_cars:
            The number of cars.

        probability:
            The probability that a random driver brakes in a step.

        safe_distance:
            The distance to the car ahead below which drivers brake.

        speed_limit:
            The largest speed.

        random_state:
            The seed or generator of the initial spacings.

    Returns:
        system:
            The model, with the group `cars` of positions `x`, speeds `v`, and ranks `rank`.

    Examples:
        >>> from skcomplex.models.engineering.traffic import charbonneau_traffic
        >>> charbonneau_traffic(5).elements[0].size
        5
    """
    rng = np.random.default_rng(random_state)
    positions = np.cumsum(rng.uniform(3.0, 17.0, n_cars))
    cars = Motorist.group(
        'cars',
        n_cars,
        x=positions,
        v=np.zeros(n_cars),
        rank=np.arange(n_cars, dtype=float),
        safe_distance=safe_distance,
        speed_limit=speed_limit,
    )
    return System(elements=[cars], interactions=[CarelessBraking('braking', cars='cars', probability=probability)])


@register_model(
    field='engineering/traffic',
    tags=('traffic', 'jams', 'agents', 'stochastic'),
    references=('Downey 2018', 'Resnick 1994'),
    defaults={'until': 100},
)
def ring_road_drivers(
    n_cars: int = 30,
    length: float = 1000.0,
    *,
    eps: float = 0.02,
    speed_limit: float = 40.0,
) -> System:
    """Return drivers on a circular highway, where small random errors of speed set off traffic jams.

    The cars start at rest, equally spaced. Every driver accelerates by 1 per step up to the speed limit, the speed
    takes a random relative error of up to `eps`, and a speed that would reach the car ahead becomes 0.

    Args:
        n_cars:
            The number of cars.

        length:
            The length of the ring road.

        eps:
            The relative amplitude of the random error of speed.

        speed_limit:
            The largest speed.

    Returns:
        system:
            The model, with the group `drivers` of speeds `speed` on a ring road of length `length`.

    Examples:
        >>> from skcomplex.models.engineering.traffic import ring_road_drivers
        >>> ring_road_drivers(10).space.width
        1000.0
    """
    positions = np.linspace(0.0, length, n_cars, endpoint=False)
    drivers = Driver.group(
        'drivers',
        n_cars,
        location=[(float(position), ROAD_WIDTH / 2) for position in positions],
        rank=np.arange(n_cars, dtype=float),
        eps=eps,
        speed_limit=speed_limit,
    )
    return System(elements=[drivers], space=ContinuousSpace(length, ROAD_WIDTH, torus=True))
