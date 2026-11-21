"""Car-following systems: cars on a road that react to the car ahead."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import cos, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import shift_member
from ....core.components.spaces import BaseSpace, ContinuousSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.traffic import CarElement, OptimalVelocityInteraction


class LeadCarInteraction(Interaction):
    """Swing of the speed of a lead car about a cruising speed `u`.

    The speed `x'` follows `u (1 - amplitude sin(omega t))`, so the acceleration is
    `x'' = -u amplitude omega cos(omega t)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the lead car.

        u:
            The cruising speed.

        amplitude:
            The relative amplitude of the swing.

        omega:
            The angular frequency of the swing.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import LeadCarInteraction
        >>> LeadCarInteraction('lead', target='car_0').omega
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        u: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        amplitude: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        omega: Annotated[float, RealSet.closed_open(0, oo)] = 0.5,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.u = u
        self.amplitude = amplitude
        self.omega = omega

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the forced swing of the speed of the lead car.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context, context.t
            self.laws_: list[BaseLaw] = [
                RateLaw(s.target.x.diff(t, 2), -s.u * s.amplitude * s.omega * cos(s.omega * t)),
            ]


class DelayedFollowingInteraction(Interaction):
    """Acceleration of a follower car by the speed difference to its leader.

    Both cars are seen a reaction time late. The speed of the follower follows the Gazis-Herman-Rothery law
    `x_f''(t) = λ (x_l' - x_f')(t - T) / (x_l - x_f)(t - T)^m` of the sensitivity `λ`, the reaction time `T`, and the
    power `m` of the gap.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the follower car.

        leader:
            The path of the car ahead of the follower.

        reaction:
            The sensitivity `λ`.

        T:
            The reaction time.

        power:
            The power `m` of the gap.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import DelayedFollowingInteraction
        >>> DelayedFollowingInteraction('follow', target='car_1', leader='car_0').T
        1.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        leader: Role,
        reaction: Annotated[float, RealSet.open(0, oo)] = 1.0,
        T: Annotated[float, RealSet.open(0, oo)] = 1.0,
        power: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.leader = leader
        self.reaction = reaction
        self.T = T
        self.power = power

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the delayed response of the follower to its leader.

        Args:
            context:
                The namespace of the cars, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context, context.t
            leader, follower = s.leader.x, s.target.x
            gap = (leader - follower)(t=t - s.T)
            approach = (leader.diff(t) - follower.diff(t))(t=t - s.T)
            self.laws_: list[BaseLaw] = [RateLaw(follower.diff(t, 2), s.reaction * approach / gap**s.power)]


class IntelligentDriverInteraction(Interaction):
    """Acceleration of every car of a ring road by the intelligent driver system.

    The acceleration depends on its speed and the car ahead. The speed `v = x'` follows
    `v' = a (1 - (v / v0)^δ - (s*(v, Δv) / (h - l))^2)`, with the desired gap `s* = s0 + v T + v Δv / (2 sqrt(a b))`,
    the approach rate `Δv` to the next car, and the headway `h` less the car length `l`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of cars, in the order they drive.

        v0:
            The desired speed.

        T:
            The safe time headway.

        a:
            The largest acceleration.

        b:
            The comfortable deceleration.

        s0:
            The jam distance.

        delta:
            The acceleration exponent.

        length:
            The length of a car.

        road_length:
            The length of the ring road.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import IntelligentDriverInteraction
        >>> IntelligentDriverInteraction('follow', target='cars').v0
        30.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        v0: Annotated[float, RealSet.open(0, oo)] = 30.0,
        T: Annotated[float, RealSet.closed_open(0, oo)] = 1.5,
        a: Annotated[float, RealSet.open(0, oo)] = 1.0,
        b: Annotated[float, RealSet.open(0, oo)] = 1.5,
        s0: Annotated[float, RealSet.closed_open(0, oo)] = 2.0,
        delta: Annotated[float, RealSet.open(0, oo)] = 4.0,
        length: Annotated[float, RealSet.closed_open(0, oo)] = 5.0,
        road_length: Annotated[float, RealSet.open(0, oo)] = 500.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.s0 = s0
        self.delta = delta
        self.length = length
        self.road_length = road_length

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of the intelligent driver system.

        Args:
            context:
                The namespace of the cars, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context, context.t
            x = s.target.x
            v = x.diff(t)
            gap = s.s0 + v * s.T + v * (v - shift_member(v, 1)) / (2 * sqrt(s.a * s.b))
            headway = OptimalVelocityInteraction._get_headway(x, s.road_length)
            self.laws_: list[BaseLaw] = [
                RateLaw(x.diff(t, 2), s.a * (1 - (v / s.v0) ** s.delta - (gap / (headway - s.length)) ** 2)),
            ]


class OptimalVelocitySystem(BaseSystem):
    """The optimal velocity system of cars on a ring road.

    Uniform flow breaks into stop-and-go waves. Car `i` relaxes its speed towards the optimal velocity of its headway
    `h`, `x_i'' = a (V(h) - x_i')` with `V(h) = V0 (tanh(h - hc) + tanh(hc))`. Uniform flow with headway
    `b = length / n_cars` is unstable when `V'(b) > a / 2`, and then a jam forms that moves against the direction of
    travel. The cars are one group, and the headway of car `i` is the distance along the ring to member `i + 1`. The
    system has the group `cars` of positions `cars.x` and speeds `cars.x'`, car `i + 1` ahead of car `i`.

    Args:
        n_cars:
            The number of cars.

        length:
            The length of the ring road.

        a:
            The sensitivity of drivers.

        V0:
            The scale of the optimal velocity, which tends to `V0 (1 + tanh(hc))` on an empty road.

        hc:
            The headway at which the optimal velocity changes fastest.

        perturbation:
            The forward shift of the first car from uniform spacing.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import OptimalVelocitySystem
        >>> system = OptimalVelocitySystem(n_cars=5).prepare().system_
        >>> system.elements[0].size, len(system.interactions)
        (5, 1)
    """

    field = 'engineering/traffic'
    topics = ('traffic', 'jams', 'ode', 'car following')
    references = ('Bando et al. 1995', 'Chowdhury, Santen, and Schadschneider 2000')
    until = 200.0
    dt = 1.0

    @staticmethod
    def _get_ring_positions(n_cars: int, length: float, perturbation: float) -> list[float]:
        """Return evenly spaced positions on a ring.

        The first car is moved forward by a perturbation.
        """
        positions = np.arange(n_cars) * length / n_cars
        positions[0] += perturbation
        return [float(position) for position in positions]

    def __init__(
        self,
        n_cars: int = 20,
        length: float = 40.0,
        *,
        a: float = 1.0,
        V0: float = 1.0,
        hc: float = 2.0,
        perturbation: float = 0.1,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_cars = n_cars
        self.length = length
        self.a = a
        self.V0 = V0
        self.hc = hc
        self.perturbation = perturbation

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        headway = self.length / self.n_cars
        speed = self.V0 * (np.tanh(headway - self.hc) + np.tanh(self.hc))
        cars = CarElement(
            'cars',
            OptimalVelocitySystem._get_ring_positions(self.n_cars, self.length, self.perturbation),
            values={"x'": float(speed)},
            size=self.n_cars,
            space=ContinuousSpace(extent=(self.length,)),
        )
        following = OptimalVelocityInteraction(
            'follow',
            target='cars',
            a=self.a,
            V0=self.V0,
            hc=self.hc,
            road_length=self.length,
        )
        self.system_ = System(self.label, elements=[cars], interactions=[following])


class IntelligentDriverSystem(BaseSystem):
    """The intelligent driver system of cars on a ring road.

    Car `i` of speed `v = x'` accelerates as `v' = a (1 - (v / v0)^δ - (s* / s)^2)` with the gap `s` to the car ahead
    and the desired gap `s* = s0 + v T + v Δv / (2 sqrt(a b))`, where `Δv` is the speed difference to the car ahead.
    Cars start at rest. The cars are one group, and the car ahead of car `i` is member `i + 1` along the ring. The
    system has the group `cars` of positions `cars.x` and speeds `cars.x'`, car `i + 1` ahead of car `i`.

    Args:
        n_cars:
            The number of cars.

        length:
            The length of the ring road.

        v0:
            The desired speed.

        T:
            The safe time headway.

        a:
            The largest acceleration.

        b:
            The comfortable deceleration.

        s0:
            The jam distance.

        delta:
            The acceleration exponent.

        car_length:
            The length of a car, subtracted from the headway to give the gap.

        perturbation:
            The forward shift of the first car from uniform spacing.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import IntelligentDriverSystem
        >>> IntelligentDriverSystem(n_cars=4).prepare().system_.interactions[0].road_length
        500.0
    """

    field = 'engineering/traffic'
    topics = ('traffic', 'jams', 'ode', 'car following')
    references = ('Treiber, Hennecke, and Helbing 2000', 'Chowdhury, Santen, and Schadschneider 2000')
    until = 100.0
    dt = 1.0

    def __init__(
        self,
        n_cars: int = 20,
        length: float = 500.0,
        *,
        v0: float = 30.0,
        T: float = 1.5,
        a: float = 1.0,
        b: float = 1.5,
        s0: float = 2.0,
        delta: float = 4.0,
        car_length: float = 5.0,
        perturbation: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_cars = n_cars
        self.length = length
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.s0 = s0
        self.delta = delta
        self.car_length = car_length
        self.perturbation = perturbation

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        cars = CarElement(
            'cars',
            OptimalVelocitySystem._get_ring_positions(self.n_cars, self.length, self.perturbation),
            size=self.n_cars,
            space=ContinuousSpace(extent=(self.length,)),
        )
        following = IntelligentDriverInteraction(
            'follow',
            target='cars',
            v0=self.v0,
            T=self.T,
            a=self.a,
            b=self.b,
            s0=self.s0,
            delta=self.delta,
            length=self.car_length,
            road_length=self.length,
        )
        self.system_ = System(self.label, elements=[cars], interactions=[following])


class DelayedCarFollowingSystem(BaseSystem):
    """A chain of cars that react to the speed difference to the car ahead.

    They react after a reaction time. Car `j + 1` follows car `j` as `x''_{j+1}(t) = λ Δv(t - T) / Δx(t - T)^power`,
    where `Δv` and `Δx` are the speed and position differences to the car ahead. The lead car drives at
    `u (1 - amplitude sin(ω t))`. Oscillations grow along the chain when `ω < 2 λ sin(ω T)` for the linear system,
    `power = 0`. The system has the lead car `car_0` and the cars `car_<j>` behind it, with positions `x` and speeds
    `x'`.

    Args:
        n_cars:
            The number of cars, the lead car included.

        spacing:
            The initial distance between neighbouring cars.

        u:
            The mean speed of the lead car.

        amplitude:
            The relative amplitude of the speed oscillation of the lead car.

        omega:
            The angular frequency of the speed oscillation of the lead car.

        reaction:
            The reaction constant `λ`.

        T:
            The reaction time.

        power:
            The power of the distance in the reaction, 0 for the linear system.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import DelayedCarFollowingSystem
        >>> DelayedCarFollowingSystem(n_cars=3).prepare().system_.elements[2].x
        -20.0
    """

    field = 'engineering/traffic'
    topics = ('traffic', 'delay', 'car following', 'instability')
    references = ('Gros 2015', 'Chandler, Herman, and Montroll 1958')
    until = 60.0
    dt = 0.5

    def __init__(
        self,
        n_cars: int = 8,
        *,
        spacing: float = 10.0,
        u: float = 1.0,
        amplitude: float = 0.1,
        omega: float = 0.5,
        reaction: float = 1.0,
        T: float = 1.0,
        power: float = 0.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_cars = n_cars
        self.spacing = spacing
        self.u = u
        self.amplitude = amplitude
        self.omega = omega
        self.reaction = reaction
        self.T = T
        self.power = power

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        cars = [
            CarElement(f'car_{index}', -index * self.spacing, values={"x'": self.u}) for index in range(self.n_cars)
        ]
        lead = LeadCarInteraction('lead', target='car_0', u=self.u, amplitude=self.amplitude, omega=self.omega)
        followers = [
            DelayedFollowingInteraction(
                f'follow_{index}',
                target=f'car_{index}',
                leader=f'car_{index - 1}',
                reaction=self.reaction,
                T=self.T,
                power=self.power,
            )
            for index in range(1, self.n_cars)
        ]
        self.system_ = System(self.label, elements=cars, interactions=[lead, *followers])
