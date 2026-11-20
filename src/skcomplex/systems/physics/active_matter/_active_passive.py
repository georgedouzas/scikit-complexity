"""Flocking of active agents among passive ones.

The agents combine self-propulsion, flocking, noise, and a soft repulsion between all.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, oo, sqrt
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import NoiseLaw, RateLaw
from ....core.components.laws.utils import neighbors
from ....core.components.spaces import BaseSpace, ContinuousSpace, RadiusNeighborhood
from ....core.engine.backends import BaseBackend, NumbaBackend
from ....core.engine.solvers import BaseSolver, EulerMaruyamaSolver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ...general.elements import PointElement

SOFTENING = 1e-12
SMALLEST_GROUP = 2
TIME_STEP = 0.01
ACTIVE_AGENT, PASSIVE_AGENT = 1.0, 0.0


class FlockingAgentElement(PointElement):
    """Agent of a crowd, a point with a mark that says whether it is active or passive.

    Args:
        label:
            The label of the agent or the group.

        x:
            The initial position.

        active:
            The mark, 1 for an active agent or 0 for a passive one.

        radius:
            The radius `r_0` of the agent, which it is drawn with.

        values:
            Further values by name, such as the starting velocity `"x'"`.

        size:
            The number of agents, or `None` for one.

        space:
            The continuous space of the agents, or `None` for none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import FlockingAgentElement
        >>> FlockingAgentElement('agent', active=0.0).active
        0.0
    """

    visual = VisualSpec(
        'points',
        {'position': 'x', 'radius': 'radius', 'color_by': 'active', 'color_range': (0.0, 1.0)},
    )

    def __init__(
        self,
        label: str,
        x: npt.ArrayLike = 0.0,
        *,
        active: Annotated[object, FiniteEnumeratedSet([PASSIVE_AGENT, ACTIVE_AGENT])] = ACTIVE_AGENT,
        radius: Annotated[float, RealSet.open(0, oo)] = 1.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, x=x, values=values, size=size, space=space)
        self.active = active
        self.radius = radius


class TargetSpeedInteraction(Interaction):
    """Propulsion towards a target speed with a random force, which only active agents feel at full strength.

    Every agent feels `mu (active v0 - |v|) v / |v|` on its velocity `v = x'`, so active agents speed up to `v0` and
    passive ones slow to rest, and a random force of amplitude `active eta`. It is a self-propelled bacterium among
    dead ones, a walking pedestrian among standing ones, or a running fan in a crowd.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the agents, whose values are `x` and `active`.

        mu:
            The strength of the propulsion.

        v0:
            The target speed of the active agents.

        eta:
            The amplitude of the random force on an active agent.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import TargetSpeedInteraction
        >>> TargetSpeedInteraction('propulsion', target='agents', v0=0.5).v0
        0.5
    """

    law_classes = (RateLaw, NoiseLaw)

    def __init__(self, label: str, *, target: Role, mu: float = 10.0, v0: float = 0.02, eta: float = 0.0) -> None:
        super().__init__(label)
        self.target = target
        self.mu = mu
        self.v0 = v0
        self.eta = eta

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the propulsion and the random force on the velocity of the agents.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context, context.t
            x = s.target.x
            velocity = x.diff(t)
            speed = sqrt(velocity.dot_product(velocity) + SOFTENING)
            self.laws_ = [
                RateLaw(x.diff(t, 2), velocity * (s.mu * (s.target.active * s.v0 - speed) / speed)),
                NoiseLaw(x.diff(t, 2), s.target.active * s.eta),
            ]


class VelocityFlockingInteraction(Interaction):
    """Flocking force of size `alpha` along the summed velocities of the neighbours, felt by active agents.

    The force is `active alpha V / |V|`, with `V` the sum of the velocities `x'` of the other agents within `reach`. It
    is the alignment of birds in a flock, of fish in a school, or of marching locusts.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the agents, whose values are `x` and `active`.

        alpha:
            The size of the flocking force.

        reach:
            The radius within which an agent reads the velocities of the others.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import VelocityFlockingInteraction
        >>> VelocityFlockingInteraction('flocking', target='agents', reach=0.2).reach
        0.2
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role, alpha: float = 1.0, reach: float = 0.1) -> None:
        super().__init__(label)
        self.target = target
        self.alpha = alpha
        self.reach = reach

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the flocking force on the velocity of the agents.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context, context.t
            x = s.target.x
            summed = neighbors(s.target.other.x.diff(t), over=RadiusNeighborhood(self.reach))
            force = summed * (s.target.active * s.alpha / sqrt(summed.dot_product(summed) + SOFTENING))
            self.laws_ = [RateLaw(x.diff(t, 2), force)]


class ActivePassiveFlockingSystem(BaseSystem):
    """The active and passive agents of Charbonneau (2017) in a periodic box.

    The system is that of section 10.4.

    All agents repel each other within `2 r_0` with the force `epsilon (1 - r / (2 r_0))^exponent`, and all feel the
    force `mu (v_t - |v|) v / |v|`, with the target speed `v_t = v0` for an active agent and 0 for a passive one. An
    active agent also feels the flocking force `alpha V / |V|`, with `V` the sum of the velocities of the agents of both
    kinds within the flocking radius, itself excluded, and a random force. Agents start at uniform random positions with
    velocity components uniform up to `initial_speed`. The defaults are those of figure 10.7 at the compactness `C = pi
    N r_0^2 / size^2 = 0.672`. The values `exponent=2.5`, `epsilon=100`, `mu=0.05`, `radius=1`, `flocking_radius=4`, and
    `v0=1` are those of Silverberg et al. (2013). The system has the group `agents` that holds the active agents first,
    the positions `agents.x`, the velocities `agents.x'`, the marks `agents.active`, and the observables
    `polarization`, `active_distance`, and `passive_distance`.

    Args:
        n_agents:
            The number of agents.

        n_active:
            The number of active agents.

        size:
            The side of the periodic box.

        radius:
            The radius `r_0` of an agent, so agents overlap below the distance `2 r_0`.

        epsilon:
            The strength of the repulsion.

        exponent:
            The exponent of the repulsion.

        flocking_radius:
            The radius within which an active agent reads the velocities of the others.

        alpha:
            The size of the flocking force on an active agent, 0 for no flocking.

        v0:
            The target speed of the active agents.

        mu:
            The strength of the propulsion.

        eta:
            The half-width of the uniform random force per step of the book. The noise is a Wiener process with the
            same variance per step, of amplitude `eta sqrt(time_step / 3)`.

        time_step:
            The step of the solver that `eta` refers to.

        initial_speed:
            The largest size of a component of an initial velocity.

        random_state:
            The seed or generator of the initial positions and velocities.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for Euler-Maruyama with the step `time_step`.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for Numba.

    Raises:
        ValidationError:
            The number of active agents is not between 1 and `n_agents`.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ActivePassiveFlockingSystem
        >>> [element.label for element in ActivePassiveFlockingSystem(114).prepare().system_.elements]
        ['agents']
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'sde', 'flocking', 'phase separation', 'pair interaction')
    references = ('Charbonneau 2017', 'Silverberg et al. 2013')
    until = 50.0
    dt = 0.5

    @staticmethod
    def _get_velocity_x(member: SimpleNamespace) -> Expression:
        """Return the first velocity component of a member when active, and 0 otherwise."""
        return member.active * member.x_d1[0]

    @staticmethod
    def _get_velocity_y(member: SimpleNamespace) -> Expression:
        """Return the second velocity component of a member when active, and 0 otherwise."""
        return member.active * member.x_d1[1]

    @staticmethod
    def _get_speed(member: SimpleNamespace) -> Expression:
        """Return the speed of a member when active, and 0 otherwise."""
        return member.active * sqrt(member.x_d1.dot_product(member.x_d1))

    @staticmethod
    def _get_polarization(s: SimpleNamespace) -> Expression:
        """Return the modulus of the mean active velocity over the mean active speed."""
        return sqrt(s.velocity_x**2 + s.velocity_y**2) / s.mean_speed

    @staticmethod
    def _get_mean_distance(t: float, values: Mapping[str, object], *, mark: float, size: float) -> float:
        """Return the mean distance between the agents of a mark in a periodic box.

        The distance is 0 for fewer than two agents.
        """
        del t
        positions = np.asarray(values['agents.x'], dtype=float)[np.asarray(values['agents.active']) == mark]
        count = len(positions)
        if count < SMALLEST_GROUP:
            return 0.0
        separations = positions[:, None, :] - positions[None, :, :]
        separations -= size * np.round(separations / size)
        distances: Array = np.sqrt((separations**2).sum(axis=-1))
        return float(distances.sum() / (count * (count - 1)))

    def __init__(
        self,
        n_agents: int = 342,
        *,
        n_active: int = 45,
        size: float = 1.0,
        radius: float = 0.025,
        epsilon: float = 25.0,
        exponent: float = 1.5,
        flocking_radius: float = 0.1,
        alpha: float = 1.0,
        v0: float = 0.02,
        mu: float = 10.0,
        eta: float = 0.1,
        time_step: float = TIME_STEP,
        initial_speed: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_agents = n_agents
        self.n_active = n_active
        self.size = size
        self.radius = radius
        self.epsilon = epsilon
        self.exponent = exponent
        self.flocking_radius = flocking_radius
        self.alpha = alpha
        self.v0 = v0
        self.mu = mu
        self.eta = eta
        self.time_step = time_step
        self.initial_speed = initial_speed
        self.random_state = random_state

    @staticmethod
    def _get_repulsion_energy(
        s: SimpleNamespace,
        r: Expression,
        *,
        epsilon: float,
        radius: float,
        exponent: float,
    ) -> Expression:
        """Return the energy of the soft repulsion between agents.

        Its force at the distance `r` is the repulsion `epsilon (1 - r / (2 r_0))^exponent`.
        """
        del s
        overlap = 1 - r / (2 * radius)
        return epsilon * 2 * radius * overlap ** (exponent + 1) / (exponent + 1)

    def _get_repulsion_laws(self, s: SimpleNamespace) -> list[BaseComponent]:
        """Return the push of every agent by the agents that overlap it.

        The push of one agent is minus the slope of the repulsion energy along the line of centres, summed over the
        agents within two radii.
        """
        x = s.target.x
        separation = x - s.target.other.x
        r = sqrt(separation.dot_product(separation))
        distance = SR.var('r', domain='positive')
        energy = self._get_repulsion_energy(
            s,
            distance,
            epsilon=self.epsilon,
            radius=self.radius,
            exponent=self.exponent,
        )
        slope = SR(energy).diff(distance).subs({distance: r})
        push = neighbors(-slope * separation / r, RadiusNeighborhood(2 * self.radius))
        return [RateLaw(x.diff(s.t, 2), push)]

    @staticmethod
    def _get_observables(system: System) -> list[Observable]:
        """Return the observables of the agents.

        They are the velocity of the active agents, their polarization, and the mean distance within each kind.
        """
        if not isinstance(system.space, ContinuousSpace):
            error_msg = (
                'The observables of `ActivePassiveFlockingSystem` read the periodic box of its continuous space.'
            )
            raise ValidationError(error_msg)
        size = float(system.space.extent[0])
        return [
            Observable('velocity_x', ActivePassiveFlockingSystem._get_velocity_x, group='agents'),
            Observable('velocity_y', ActivePassiveFlockingSystem._get_velocity_y, group='agents'),
            Observable('mean_speed', ActivePassiveFlockingSystem._get_speed, group='agents'),
            Observable('polarization', ActivePassiveFlockingSystem._get_polarization),
            Observable(
                'active_distance',
                function=functools.partial(
                    ActivePassiveFlockingSystem._get_mean_distance,
                    mark=ACTIVE_AGENT,
                    size=size,
                ),
            ),
            Observable(
                'passive_distance',
                function=functools.partial(
                    ActivePassiveFlockingSystem._get_mean_distance,
                    mark=PASSIVE_AGENT,
                    size=size,
                ),
            ),
        ]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the active and passive agents at random places and velocities in a periodic box.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.

        Raises:
            ValidationError:
                The number of active agents is not between 1 and `n_agents`.
        """
        super()._prepare(context)
        if not 1 <= self.n_active <= self.n_agents:
            error_msg = (
                f'The system has {self.n_active} active agents among {self.n_agents}. Give at least one, and at most '
                'all.'
            )
            raise ValidationError(error_msg)
        rng = np.random.default_rng(self.random_state)
        box = ContinuousSpace(extent=(self.size, self.size), boundary='periodic')
        agents = FlockingAgentElement(
            'agents',
            x=rng.uniform(0.0, self.size, (self.n_agents, 2)),
            active=np.where(np.arange(self.n_agents) < self.n_active, ACTIVE_AGENT, PASSIVE_AGENT),
            radius=self.radius,
            values={"x'": rng.uniform(-self.initial_speed, self.initial_speed, (self.n_agents, 2))},
            size=self.n_agents,
            space=box,
        )
        interactions = [
            Interaction('repulsion', roles={'target': 'agents'}, laws=self._get_repulsion_laws),
            TargetSpeedInteraction(
                'propulsion',
                target='agents',
                mu=self.mu,
                v0=self.v0,
                eta=self.eta * math.sqrt(self.time_step / 3),
            ),
            VelocityFlockingInteraction('flocking', target='agents', alpha=self.alpha, reach=self.flocking_radius),
        ]
        self.system_ = System(
            self.label,
            elements=[agents],
            interactions=interactions,
            space=box,
            solver=EulerMaruyamaSolver(dt=self.time_step),
            backend=NumbaBackend(),
        )
        self.observables_ = tuple(self._get_observables(self.system_))
