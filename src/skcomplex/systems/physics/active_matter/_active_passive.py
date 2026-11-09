"""Flocking of active agents among passive ones.

The agents combine self-propulsion, flocking, noise, and a soft repulsion between all.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import math
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, sqrt
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components.elements import GroupElement, PointElement
from ....core.components.interactions import PairInteraction, RateInteraction
from ....core.components.laws.utils import neighbors
from ....core.components.spaces import ContinuousSpace, RadiusNeighborhood
from ....core.components.systems import BaseSystem, CompositeSystem, RunSettings
from ....core.engine.backends import NumbaBackend
from ....core.engine.solvers import EulerMaruyamaSolver
from ....core.errors import ValidationError
from ....core.specs import ObservableSpec, StateSpec, VisualSpec
from ....core.types import Array

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

        x_d1:
            The initial velocity.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        active:
            The mark, 1 for an active agent or 0 for a passive one.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import FlockingAgentElement
        >>> FlockingAgentElement('agent', active=0.0).active
        0.0
    """

    def __init__(
        self,
        label: str,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, order=2, role='position')] = 0.0,
        x_d1: npt.ArrayLike = 0.0,
        visual: VisualSpec | None = None,
        *,
        active: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([PASSIVE_AGENT, ACTIVE_AGENT]))] = ACTIVE_AGENT,
    ) -> None:
        super().__init__(label, x=x, x_d1=x_d1, visual=visual)
        self.active = active


def _velocity_x(member: SimpleNamespace) -> Expression:
    """Return the first velocity component of a member when active, and 0 otherwise."""
    return member.active * member.x_d1[0]


def _velocity_y(member: SimpleNamespace) -> Expression:
    """Return the second velocity component of a member when active, and 0 otherwise."""
    return member.active * member.x_d1[1]


def _speed(member: SimpleNamespace) -> Expression:
    """Return the speed of a member when active, and 0 otherwise."""
    return member.active * sqrt(member.x_d1.dot_product(member.x_d1))


def _polarization(s: SimpleNamespace) -> Expression:
    """Return the modulus of the mean active velocity over the mean active speed."""
    return sqrt(s.velocity_x**2 + s.velocity_y**2) / s.mean_speed


def _mean_distance(t: float, values: Mapping[str, object], *, mark: float, size: float) -> float:
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
    the positions `agents.x`, the velocities `agents.x_d1`, the marks `agents.active`, and the observables
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
    solver = EulerMaruyamaSolver(dt=TIME_STEP)
    backend = NumbaBackend()

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
    ) -> None:
        super().__init__(label)
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
    def _repulsion_energy(
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

    @staticmethod
    def _propulsion(s: SimpleNamespace, *, mu: float, v0: float) -> FreeModuleElement:
        """Return the force `mu (v_t - |v|) v / |v|` towards the target speed `v_t`.

        The target speed is `v0` when active and 0 when passive.
        """
        velocity = s.target.x_d1
        speed = sqrt(velocity.dot_product(velocity) + SOFTENING)
        return velocity * (mu * (s.target.active * v0 - speed) / speed)

    @staticmethod
    def _flocking(s: SimpleNamespace, *, alpha: float, reach: float) -> FreeModuleElement:
        """Return the flocking force of size `alpha` on an active agent.

        The force points along the summed velocities of the others in reach.
        """
        summed = neighbors(s.target.other.x_d1, over=RadiusNeighborhood(reach))
        return summed * (s.target.active * alpha / sqrt(summed.dot_product(summed) + SOFTENING))

    @staticmethod
    def _random_force(s: SimpleNamespace, *, amplitude: float) -> Expression:
        """Return the amplitude of the noise of an agent, 0 when passive."""
        return s.target.active * amplitude

    @staticmethod
    def _observables(system: CompositeSystem) -> list[ObservableSpec]:
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
            ObservableSpec('velocity_x', _velocity_x, group='agents'),
            ObservableSpec('velocity_y', _velocity_y, group='agents'),
            ObservableSpec('mean_speed', _speed, group='agents'),
            ObservableSpec('polarization', _polarization),
            ObservableSpec('active_distance', function=functools.partial(_mean_distance, mark=ACTIVE_AGENT, size=size)),
            ObservableSpec(
                'passive_distance',
                function=functools.partial(_mean_distance, mark=PASSIVE_AGENT, size=size),
            ),
        ]

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings.

        The observables of the parts are built from the model parameters.
        """
        settings = super().get_run_settings()
        settings['observables'] = tuple(self._observables(self._build_system()))
        return settings

    def _build_system(self) -> CompositeSystem:
        """Return the active and passive agents of Charbonneau (2017) in a periodic box.

        The system is that of section 10.4.
        """
        if not 1 <= self.n_active <= self.n_agents:
            error_msg = (
                f'The system has {self.n_active} active agents among {self.n_agents}. Give at least one, and at most '
                'all.'
            )
            raise ValidationError(error_msg)
        rng = np.random.default_rng(self.random_state)
        agents = GroupElement(
            'agents',
            element=FlockingAgentElement,
            size=self.n_agents,
            values={
                'x': rng.uniform(0.0, self.size, (self.n_agents, 2)),
                'x_d1': rng.uniform(-self.initial_speed, self.initial_speed, (self.n_agents, 2)),
                'active': np.where(np.arange(self.n_agents) < self.n_active, ACTIVE_AGENT, PASSIVE_AGENT),
            },
            visual=VisualSpec(
                'points',
                {'position': 'x', 'radius': self.radius, 'color_by': 'active', 'color_range': (0.0, 1.0)},
            ),
        )
        energy = functools.partial(
            self._repulsion_energy,
            epsilon=self.epsilon,
            radius=self.radius,
            exponent=self.exponent,
        )
        return CompositeSystem(
            self.label,
            elements=[agents],
            interactions=[
                PairInteraction(
                    'repulsion',
                    'agents',
                    'agents',
                    cutoff=2 * self.radius,
                    potential=energy,
                    period=self.size,
                ),
                RateInteraction(
                    'propulsion',
                    target='agents',
                    rate=functools.partial(self._propulsion, mu=self.mu, v0=self.v0),
                    noise=functools.partial(self._random_force, amplitude=self.eta * math.sqrt(self.time_step / 3)),
                ),
                RateInteraction(
                    'flocking',
                    target='agents',
                    rate=functools.partial(self._flocking, alpha=self.alpha, reach=self.flocking_radius),
                ),
            ],
            space=ContinuousSpace(extent=(self.size, self.size), boundary='periodic'),
        )
