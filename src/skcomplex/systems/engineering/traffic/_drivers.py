"""Driver agents: cars whose drivers choose their speed.

The speed depends on the distance to the car ahead.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, cast

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace, ContinuousSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.errors import SimulationError
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Role

ROAD_WIDTH = 1.0


class MotoristView(AgentViewProtocol, Protocol):
    """View of a car of Charbonneau's traffic system.

    Its states and parameters are attributes.
    """

    x: float
    v: float
    rank: float
    safe_distance: float
    braking: float
    acceleration: float
    speed_limit: float


class MotoristElement(Element):
    """Car of Charbonneau's traffic system.

    It brakes when it is closer than a safe distance to the car ahead, by the rule of `SafeDistanceInteraction`.

    Args:
        label:
            The label of the agent or group.

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

        size:
            The number of cars, or `None` for one.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import MotoristElement
        >>> MotoristElement('cars', x=[0.0, 10.0, 20.0], size=3).size
        3
    """

    visual = VisualSpec('agents', {'position': 'x', 'color_by': 'v'})

    def __init__(
        self,
        label: str,
        *,
        x: object = 0.0,
        v: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        rank: Annotated[object, NN] = 0.0,
        safe_distance: Annotated[float, RealSet.open(0, oo)] = 5.0,
        braking: Annotated[float, RealSet.closed_open(0, oo)] = 3.0,
        acceleration: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        speed_limit: Annotated[float, RealSet.open(0, oo)] = 10.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.x = x
        self.v = v
        self.rank = rank
        self.safe_distance = safe_distance
        self.braking = braking
        self.acceleration = acceleration
        self.speed_limit = speed_limit


class SafeDistanceInteraction(Interaction):
    """Speed rule of the cars of Charbonneau's traffic system.

    Every step a driver brakes by its braking when the car ahead is closer than its safe distance, and accelerates by
    its acceleration otherwise, within its speed limit.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of cars.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import SafeDistanceInteraction
        >>> SafeDistanceInteraction('adjust', target='cars').target
        'cars'
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the speed rule of every car."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(context.target, self.adjust)]

    def adjust(self, agent: MotoristView, system: SystemView) -> None:
        """Brake when the car ahead is too close and accelerate otherwise.

        The car brakes when the car ahead is closer than the safe distance. The lead car, which has no car ahead, keeps
        its distance to the car behind it instead.

        Args:
            agent:
                The car.

            system:
                The system.
        """
        cars = cast(list[MotoristView], system.get_agents(self.target))
        ahead = next((car for car in cars if int(car.rank) == int(agent.rank) + 1), None)
        if ahead is None:
            behind = next((car for car in cars if int(car.rank) == int(agent.rank) - 1), None)
            distance = agent.x - behind.x if behind is not None else np.inf
        else:
            distance = ahead.x - agent.x
        change = -agent.braking if distance < agent.safe_distance else agent.acceleration
        agent.v = float(np.clip(agent.v + change, 0.0, agent.speed_limit))


class DriverView(AgentViewProtocol, Protocol):
    """View of a driver on a ring road, with its states and parameters as attributes."""

    speed: float
    rank: float
    acceleration: float
    min_acc: float
    max_acc: float
    speed_limit: float
    eps: float


class DriverElement(Element):
    """Driver on a ring road who accelerates as much as allowed.

    The driver stops when the speed would cause a collision, by the rule of `RingRoadDrivingInteraction`.

    Args:
        label:
            The label of the agent or group.

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

        values:
            Further values by name, such as the position `x` and `y` of every driver on the road.

        size:
            The number of drivers, or `None` for one.

        space:
            The road, a continuous space whose axes `x` and `y` place the drivers, or `None` for none.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import DriverElement
        >>> DriverElement('drivers', values={'x': [0.0, 10.0], 'y': 0.5}, size=2).size
        2
    """

    visual = VisualSpec('agents', {'color_by': 'speed'})

    def __init__(
        self,
        label: str,
        *,
        speed: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        rank: Annotated[object, NN] = 0.0,
        acceleration: float = 1.0,
        min_acc: float = -10.0,
        max_acc: float = 1.0,
        speed_limit: Annotated[float, RealSet.open(0, oo)] = 40.0,
        eps: Annotated[float, RealSet.closed_open(0, oo)] = 0.02,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.speed = speed
        self.rank = rank
        self.acceleration = acceleration
        self.min_acc = min_acc
        self.max_acc = max_acc
        self.speed_limit = speed_limit
        self.eps = eps


class RingRoadDrivingInteraction(Interaction):
    """Driving rule of the drivers on a ring road.

    The speed of a driver changes by its acceleration, bounded by `min_acc` and `max_acc`, is multiplied by a random
    factor between `1 - eps` and `1 + eps`, and stays between 0 and the speed limit. A speed larger than the distance to
    the car ahead becomes 0. The car then moves along the ring by its speed, and the ring is as long as the width of
    the space.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of drivers.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import RingRoadDrivingInteraction
        >>> RingRoadDrivingInteraction('drive', target='drivers').target
        'drivers'
    """

    draws = True

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the driving rule of every driver."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(context.target, self.drive)]

    def drive(self, agent: DriverView, system: SystemView) -> None:
        """Choose the speed from the distance to the car ahead and move along the ring.

        Args:
            agent:
                The driver.

            system:
                The system.
        """
        cars = cast(list[DriverView], system.get_agents(self.target))
        if system.dimensions is None:
            error_msg = 'Drivers move along a ring road, a continuous space with an extent.'
            raise SimulationError(error_msg)
        length = system.dimensions[0]
        ahead = self._get_car_ahead(agent, system, len(cars))
        x, y = RingRoadDrivingInteraction._get_position(agent)
        distance = (RingRoadDrivingInteraction._get_position(ahead)[0] - x) % length if ahead is not None else length
        acceleration = min(max(agent.acceleration, agent.min_acc), agent.max_acc)
        speed = (agent.speed + acceleration) * system.rng.uniform(1 - agent.eps, 1 + agent.eps)
        speed = min(max(speed, 0.0), agent.speed_limit)
        agent.speed = 0.0 if speed > distance else speed
        agent.move_to(((x + agent.speed) % length, y))

    @staticmethod
    def _get_position(agent: AgentViewProtocol) -> tuple[float, float]:
        """Return the position of an agent in a continuous space."""
        x, y = np.asarray(agent.location, dtype=float)
        return float(x), float(y)

    def _get_car_ahead(self, agent: DriverView, system: SystemView, n_cars: int) -> DriverView | None:
        """Return the car whose rank follows the agent's rank on a road of `n_cars` cars.

        It is `None` for the last car.
        """
        rank = (int(agent.rank) + 1) % n_cars if n_cars else 0
        return next(
            (
                car
                for car in cast(list[DriverView], system.get_agents(self.target))
                if int(car.rank) == rank and car != agent
            ),
            None,
        )


class CarelessBrakingInteraction(Interaction):
    """System rule of Charbonneau's traffic system.

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
        >>> from skcomplex.systems.engineering.traffic import CarelessBrakingInteraction
        >>> CarelessBrakingInteraction('bozo', cars='cars').probability
        0.1
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        cars: Role,
        probability: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        gap: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.cars = cars
        self.probability = probability
        self.gap = gap

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rule of random braking and motion of the whole road."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(context, self.drive)]

    def drive(self, system: SystemView) -> None:
        """Make a random driver brake, then move every car.

        The driver brakes with the interaction's probability.

        Args:
            system:
                The system.
        """
        cars = sorted(cast(list[MotoristView], system.get_agents('cars')), key=lambda car: car.rank)
        if not cars:
            return
        if system.rng.random() < self.probability:
            careless = cars[int(system.rng.integers(len(cars)))]
            careless.v = max(careless.v - careless.braking, 0.0)
        gap = self.gap
        positions = [car.x for car in cars]
        for index, car in enumerate(cars):
            moved = car.x + car.v
            car.x = min(moved, positions[index + 1] - gap) if index + 1 < len(cars) else moved


class CharbonneauTrafficSystem(BaseSystem):
    """Charbonneau's traffic system.

    Random braking of single drivers sets off traffic jams. Cars on a one-way road start at rest with random spacings
    between 3 and 17. Every step, each driver brakes by 3 when the car ahead is closer than the safe distance and
    accelerates by 1 otherwise, within the speed limit. A random driver brakes with a probability every step, and then
    every car moves by its speed. The system has the group `cars` of positions `x`, speeds `v`, and ranks `rank`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import CharbonneauTrafficSystem
        >>> CharbonneauTrafficSystem(5).prepare().system_.elements[0].size
        5
    """

    field = 'engineering/traffic'
    topics = ('traffic', 'jams', 'agents', 'avalanches', 'stochastic')
    references = ('Charbonneau 2017',)
    until = 100

    def __init__(
        self,
        n_cars: int = 100,
        *,
        probability: float = 0.1,
        safe_distance: float = 5.0,
        speed_limit: float = 10.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_cars = n_cars
        self.probability = probability
        self.safe_distance = safe_distance
        self.speed_limit = speed_limit
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        positions = np.cumsum(rng.uniform(3.0, 17.0, self.n_cars))
        cars = MotoristElement(
            'cars',
            x=positions,
            v=np.zeros(self.n_cars),
            rank=np.arange(self.n_cars, dtype=float),
            safe_distance=self.safe_distance,
            speed_limit=self.speed_limit,
            size=self.n_cars,
        )
        self.system_ = System(
            self.label,
            elements=[cars],
            interactions=[
                SafeDistanceInteraction('adjust', target='cars'),
                CarelessBrakingInteraction('braking', cars='cars', probability=self.probability),
            ],
        )


class RingRoadDriversSystem(BaseSystem):
    """Drivers on a circular highway.

    Small random errors of speed set off traffic jams. The cars start at rest, equally spaced. Every driver accelerates
    by 1 per step up to the speed limit, the speed takes a random relative error of up to `eps`, and a speed that would
    reach the car ahead becomes 0. The system has the group `drivers` of speeds `speed` on a ring road of length
    `length`.

    Args:
        n_cars:
            The number of cars.

        length:
            The length of the ring road.

        eps:
            The relative amplitude of the random error of speed.

        speed_limit:
            The largest speed.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import RingRoadDriversSystem
        >>> RingRoadDriversSystem(10).prepare().system_.space.prepare().extent_[0]
        1000.0
    """

    field = 'engineering/traffic'
    topics = ('traffic', 'jams', 'agents', 'stochastic')
    references = ('Downey 2018', 'Resnick 1994')
    until = 100

    def __init__(
        self,
        n_cars: int = 30,
        length: float = 1000.0,
        *,
        eps: float = 0.02,
        speed_limit: float = 40.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_cars = n_cars
        self.length = length
        self.eps = eps
        self.speed_limit = speed_limit

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        positions = np.linspace(0.0, self.length, self.n_cars, endpoint=False)
        road = ContinuousSpace(extent=(self.length, ROAD_WIDTH), boundary='periodic')
        drivers = DriverElement(
            'drivers',
            rank=np.arange(self.n_cars, dtype=float),
            speed_limit=self.speed_limit,
            eps=self.eps,
            values={'x': positions, 'y': ROAD_WIDTH / 2},
            size=self.n_cars,
            space=road,
        )
        self.system_ = System(
            self.label,
            elements=[drivers],
            interactions=[RingRoadDrivingInteraction('drive', target='drivers')],
            space=road,
        )
