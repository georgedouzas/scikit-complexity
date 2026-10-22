"""Active particle models: active Brownian, chiral, Ornstein-Uhlenbeck, run-and-tumble, and depot-pumped particles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import cos, sin, sqrt

from ....core.base import Domain, VisualSpec, parameter, state, vector
from ....core.model import ContinuousSpace, Element, Law, Noise, Rate, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, EulerMaruyama, ModelView

type Array = npt.NDArray[np.float64]

POINTS = VisualSpec('points', {'position': 'x', 'radius': 0.2})


class ActiveParticle(Element):
    """Self-propelled particle in the plane, `dx = v (cos theta, sin theta) dt + sqrt(2 D_t) dW`.

    The orientation turns at the angular velocity `omega` and diffuses, `dtheta = omega dt + sqrt(2 D_r) dW`. With
    `omega = 0` it is an active Brownian particle, and otherwise a chiral active particle.

    Args:
        label:
            The label of the particle or the group.

        x:
            The initial position.

        theta:
            The initial orientation.

        speed:
            The self-propulsion speed.

        D_t:
            The translational diffusion coefficient.

        D_r:
            The rotational diffusion coefficient.

        omega:
            The angular velocity of a chiral particle.

    Examples:
        >>> from skcomplex.models.physics.active_matter import ActiveParticle
        >>> ActiveParticle('swimmer', speed=2.0).speed
        2.0
    """

    visual: ClassVar[VisualSpec] = POINTS
    x: npt.ArrayLike = state(default=0.0, vector=True, position=True)
    theta: float = state(default=0.0, domain=Domain.ANGLE)
    speed: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    D_t: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    D_r: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    omega: float = parameter(default=0.0)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the self-propulsion, the turning, and the translational and rotational noise.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            Rate(s.x, s.speed * vector([cos(s.theta), sin(s.theta)])),
            Noise(s.x, sqrt(2 * s.D_t)),
            Rate(s.theta, s.omega),
            Noise(s.theta, sqrt(2 * s.D_r)),
        ]


class OrnsteinUhlenbeckParticle(Element):
    """Active Ornstein-Uhlenbeck particle, whose propulsion velocity `u` is an Ornstein-Uhlenbeck process.

    The laws are `dx = u dt + sqrt(2 D_t) dW` and `tau du = -u dt + sqrt(2 D) dW`, so the propulsion keeps its
    direction for the persistence time `tau` and the particle diffuses with `D + D_t` at long times.

    Args:
        label:
            The label of the particle or the group.

        x:
            The initial position.

        u:
            The initial propulsion velocity.

        tau:
            The persistence time.

        D:
            The active diffusion coefficient.

        D_t:
            The translational diffusion coefficient.

    Examples:
        >>> from skcomplex.models.physics.active_matter import OrnsteinUhlenbeckParticle
        >>> OrnsteinUhlenbeckParticle('swimmer').tau
        1.0
    """

    visual: ClassVar[VisualSpec] = POINTS
    x: npt.ArrayLike = state(default=0.0, vector=True)
    u: npt.ArrayLike = state(default=0.0, vector=True)
    tau: float = parameter(default=1.0, domain=Domain.POSITIVE)
    D: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    D_t: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the propulsion, its relaxation and noise, and the translational noise.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            Rate(s.x, s.u),
            Noise(s.x, sqrt(2 * s.D_t)),
            Rate(s.u, -s.u / s.tau),
            Noise(s.u, sqrt(2 * s.D) / s.tau),
        ]


class DepotParticle(Element):
    """Brownian particle pumped by an internal energy depot, as in the model of Schweitzer, Ebeling, and Tilch.

    The depot takes energy at the rate `q`, loses it at the rate `c e`, and converts it into motion at the rate
    `d e |v|^2`, so `de/dt = q - c e - d e |v|^2` and `dv = (-gamma v + d e v) dt + sqrt(2 D) dW`. Above
    `q d = c gamma` the particle moves at the speed `sqrt(q / gamma - c / d)`.

    Args:
        label:
            The label of the particle or the group.

        x:
            The initial position.

        v:
            The initial velocity.

        e:
            The initial energy of the depot.

        gamma:
            The friction coefficient.

        q:
            The rate of energy intake.

        c:
            The rate of internal energy loss.

        d:
            The rate of conversion of depot energy into motion.

        D:
            The intensity of the velocity noise.

    Examples:
        >>> from skcomplex.models.physics.active_matter import DepotParticle
        >>> DepotParticle('walker').q
        10.0
    """

    visual: ClassVar[VisualSpec] = POINTS
    x: npt.ArrayLike = state(default=0.0, vector=True)
    v: npt.ArrayLike = state(default=0.0, vector=True)
    e: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    gamma: float = parameter(default=20.0, domain=Domain.POSITIVE)
    q: float = parameter(default=10.0, domain=Domain.NONNEGATIVE)
    c: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    d: float = parameter(default=10.0, domain=Domain.NONNEGATIVE)
    D: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the motion, the friction and pumping of the velocity, its noise, and the depot balance.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            Rate(s.x, s.v),
            Rate(s.v, (-s.gamma + s.d * s.e) * s.v),
            Noise(s.v, sqrt(2 * s.D)),
            Rate(s.e, s.q - s.c * s.e - s.d * s.e * s.v.dot_product(s.v)),
        ]


class Tumbler(Element):
    """Run-and-tumble agent in a periodic square, which runs straight and tumbles to a random direction.

    Every step the agent moves `speed` along its direction, then tumbles with probability `1 - exp(-tumble_rate)`
    to a direction drawn uniformly, as bacteria such as E. coli do.

    Args:
        label:
            The label of the agent or group.

        theta:
            The initial direction.

        speed:
            The distance run per step.

        tumble_rate:
            The rate of tumbles per step.

        width:
            The side of the periodic square.

    Examples:
        >>> from skcomplex.models.physics.active_matter import Tumbler
        >>> Tumbler('bacterium').speed
        0.1
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'theta'})
    theta: float = state(default=0.0)
    speed: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    tumble_rate: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    width: float = parameter(default=10.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the run-and-tumble rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.run_and_tumble)]

    def run_and_tumble(self, agent: AgentView, model: ModelView) -> None:
        """Run one step along the direction, then tumble at random.

        Args:
            agent:
                The agent.

            model:
                The model.
        """
        position = np.asarray(agent.location, dtype=float)
        heading = np.array([math.cos(agent.theta), math.sin(agent.theta)])
        agent.move_to(np.mod(position + agent.speed * heading, agent.width))
        if model.rng.random() < -math.expm1(-agent.tumble_rate):
            agent.theta = float(model.rng.uniform(0.0, 2 * math.pi))


def _swimmers(
    n_particles: int,
    size: float,
    random_state: int | np.random.Generator | None,
) -> tuple[Array, Array]:
    """Return uniform positions in a square and uniform orientations."""
    rng = np.random.default_rng(random_state)
    return rng.uniform(0.0, size, (n_particles, 2)), rng.uniform(0.0, 2 * math.pi, n_particles)


@register_model(
    field='physics/active_matter',
    tags=('active matter', 'sde', 'self-propulsion', 'diffusion'),
    references=('Argun et al. 2021', 'Bechinger et al. 2016', 'Volpe, Gigan, and Volpe 2014'),
    defaults={'until': 20.0, 'dt': 0.5, 'solver': EulerMaruyama(dt=0.01)},
)
def active_brownian_particles(
    n_particles: int = 200,
    *,
    speed: float = 3.0,
    D_t: float = 0.1,
    D_r: float = 1.0,
    size: float = 50.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return active Brownian particles in the plane, which swim at a constant speed in a diffusing direction.

    Their mean squared displacement is ballistic below the persistence time `1 / D_r` and diffusive above it, with
    the effective diffusion coefficient `D_t + v^2 / (2 D_r)`.

    Args:
        n_particles:
            The number of particles.

        speed:
            The self-propulsion speed.

        D_t:
            The translational diffusion coefficient.

        D_r:
            The rotational diffusion coefficient.

        size:
            The side of the square in which the particles start.

        random_state:
            The seed or generator of the initial positions and orientations.

    Returns:
        system:
            The model, with the positions `swimmers.x` and orientations `swimmers.theta`.

    Examples:
        >>> from skcomplex.models.physics.active_matter import active_brownian_particles
        >>> active_brownian_particles(10).elements[0].size
        10
    """
    positions, angles = _swimmers(n_particles, size, random_state)
    group = ActiveParticle.group('swimmers', n_particles, x=positions, theta=angles, speed=speed, D_t=D_t, D_r=D_r)
    return System(components=[group], space=ContinuousSpace(n_dim=2))


@register_model(
    field='physics/active_matter',
    tags=('active matter', 'sde', 'self-propulsion', 'chirality'),
    references=('Bechinger et al. 2016', 'van Teeffelen and Löwen 2008'),
    defaults={'until': 20.0, 'dt': 0.5, 'solver': EulerMaruyama(dt=0.01)},
)
def chiral_active_particles(
    n_particles: int = 200,
    *,
    speed: float = 3.0,
    omega: float = 2.0,
    D_t: float = 0.1,
    D_r: float = 0.1,
    size: float = 50.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return chiral active particles, which swim along circles of radius `v / omega` that drift by diffusion.

    Their effective diffusion coefficient at long times is `D_t + v^2 D_r / (2 (D_r^2 + omega^2))`.

    Args:
        n_particles:
            The number of particles.

        speed:
            The self-propulsion speed.

        omega:
            The angular velocity.

        D_t:
            The translational diffusion coefficient.

        D_r:
            The rotational diffusion coefficient.

        size:
            The side of the square in which the particles start.

        random_state:
            The seed or generator of the initial positions and orientations.

    Returns:
        system:
            The model, with the positions `swimmers.x` and orientations `swimmers.theta`.

    Examples:
        >>> from skcomplex.models.physics.active_matter import chiral_active_particles
        >>> chiral_active_particles(10, omega=1.0).elements[0].values['omega']
        1.0
    """
    positions, angles = _swimmers(n_particles, size, random_state)
    group = ActiveParticle.group(
        'swimmers',
        n_particles,
        x=positions,
        theta=angles,
        speed=speed,
        omega=omega,
        D_t=D_t,
        D_r=D_r,
    )
    return System(components=[group], space=ContinuousSpace(n_dim=2))


@register_model(
    field='physics/active_matter',
    tags=('active matter', 'sde', 'self-propulsion', 'colored noise'),
    references=('Bechinger et al. 2016', 'Fodor et al. 2016'),
    defaults={'until': 20.0, 'dt': 0.5, 'solver': EulerMaruyama(dt=0.01)},
)
def active_ornstein_uhlenbeck_particles(
    n_particles: int = 200,
    *,
    tau: float = 1.0,
    D: float = 1.0,
    D_t: float = 0.0,
    size: float = 50.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return active Ornstein-Uhlenbeck particles in the plane, driven by exponentially correlated propulsion.

    The propulsion starts in its stationary distribution, normal with variance `D / tau` per component.

    Args:
        n_particles:
            The number of particles.

        tau:
            The persistence time.

        D:
            The active diffusion coefficient.

        D_t:
            The translational diffusion coefficient.

        size:
            The side of the square in which the particles start.

        random_state:
            The seed or generator of the initial positions and propulsions.

    Returns:
        system:
            The model, with the positions `swimmers.x` and propulsions `swimmers.u`.

    Examples:
        >>> from skcomplex.models.physics.active_matter import active_ornstein_uhlenbeck_particles
        >>> active_ornstein_uhlenbeck_particles(10, tau=0.5).elements[0].values['tau']
        0.5
    """
    rng = np.random.default_rng(random_state)
    positions = rng.uniform(0.0, size, (n_particles, 2))
    propulsion = rng.normal(0.0, math.sqrt(D / tau), (n_particles, 2))
    group = OrnsteinUhlenbeckParticle.group('swimmers', n_particles, x=positions, u=propulsion, tau=tau, D=D, D_t=D_t)
    return System(components=[group], space=ContinuousSpace(n_dim=2))


@register_model(
    field='physics/active_matter',
    tags=('active matter', 'agents', 'bacteria', 'self-propulsion'),
    references=('Berg 2004', 'Bechinger et al. 2016'),
    defaults={'until': 100, 'dt': 1},
)
def run_and_tumble(
    n_agents: int = 100,
    *,
    speed: float = 0.1,
    tumble_rate: float = 0.1,
    size: float = 10.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return run-and-tumble agents in a periodic square, which diffuse with `v^2 (1 + q) / (4 (1 - q))` per step.

    The persistence `q = exp(-tumble_rate)` is the probability that an agent keeps its direction for one step.

    Args:
        n_agents:
            The number of agents.

        speed:
            The distance run per step.

        tumble_rate:
            The rate of tumbles per step.

        size:
            The side of the periodic square.

        random_state:
            The seed or generator of the initial positions and directions.

    Returns:
        system:
            The model, with the agent group `bacteria`.

    Examples:
        >>> from skcomplex.models.physics.active_matter import run_and_tumble
        >>> run_and_tumble(10).elements[0].size
        10
    """
    positions, angles = _swimmers(n_agents, size, random_state)
    group = Tumbler.group(
        'bacteria',
        n_agents,
        location=positions,
        theta=angles,
        speed=speed,
        tumble_rate=tumble_rate,
        width=size,
    )
    return System(components=[group], space=ContinuousSpace(size, size, boundary='periodic'))


@register_model(
    field='physics/active_matter',
    tags=('active matter', 'sde', 'energy depot', 'self-propulsion'),
    references=('Schweitzer, Ebeling, and Tilch 1998', 'Helbing 2001'),
    defaults={'until': 10.0, 'dt': 0.1, 'solver': EulerMaruyama(dt=0.001)},
)
def depot_brownian_particles(
    n_particles: int = 50,
    *,
    gamma: float = 20.0,
    q: float = 10.0,
    c: float = 1.0,
    d: float = 10.0,
    D: float = 0.01,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Brownian particles with energy depots, which start at rest and take up the speed `sqrt(q/gamma - c/d)`.

    Args:
        n_particles:
            The number of particles.

        gamma:
            The friction coefficient.

        q:
            The rate of energy intake.

        c:
            The rate of internal energy loss.

        d:
            The rate of conversion of depot energy into motion.

        D:
            The intensity of the velocity noise.

        random_state:
            The seed or generator of the small initial velocities.

    Returns:
        system:
            The model, with the positions `walkers.x`, velocities `walkers.v`, and depots `walkers.e`.

    Examples:
        >>> from skcomplex.models.physics.active_matter import depot_brownian_particles
        >>> depot_brownian_particles(5).elements[0].values['q']
        10.0
    """
    velocities = np.random.default_rng(random_state).normal(0.0, 0.01, (n_particles, 2))
    group = DepotParticle.group('walkers', n_particles, v=velocities, gamma=gamma, q=q, c=c, d=d, D=D)
    return System(components=[group], space=ContinuousSpace(n_dim=2))
