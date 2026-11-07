"""Active particle systems: active Brownian, chiral, Ornstein-Uhlenbeck, run-and-tumble, and depot-pumped particles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, cos, oo, sin, sqrt, vector
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw, NoiseLaw, RateLaw, RuleLaw
from ....core.components.spaces import ContinuousSpace
from ....core.components.system import System
from ....core.engine.solvers import AgentView, EulerMaruyamaSolver, SystemView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..._catalog import register_system
from ..common.particles import POINTS, ActiveParticleElement, SelfPropulsionInteraction


class DepotParticleElement(Element):
    """Brownian particle pumped by an internal energy depot, as in the system of Schweitzer, Ebeling, and Tilch.

    The depot takes energy at the rate `q`, loses it at the rate `c e`, and converts it into motion at the rate
    `d e |v|^2`, so `de/dt = q - c e - d e |v|^2` and `dv = (-gamma v + d e v) dt + sqrt(2 D) dW`. Above
    `q d = c gamma` the particle moves at the speed `sqrt(q / gamma - c / d)`.

    Args:
        label:
            The label of the particle or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.systems.physics.active_matter import DepotParticleElement
        >>> DepotParticleElement('walker').q
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = POINTS,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
        v: Annotated[npt.ArrayLike, StateSpec(vector=True, role='velocity')] = 0.0,
        e: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20.0,
        q: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        c: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v
        self.e = e
        self.gamma = gamma
        self.q = q
        self.c = c
        self.d = d
        self.D = D

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the motion, the friction and pumping of the velocity, its noise, and the depot balance.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            RateLaw(s.x, s.v),
            RateLaw(s.v, (-s.gamma + s.d * s.e) * s.v),
            NoiseLaw(s.v, sqrt(2 * s.D)),
            RateLaw(s.e, s.q - s.c * s.e - s.d * s.e * s.v.dot_product(s.v)),
        ]


class InertialActiveParticleElement(Element):
    """Active Brownian particle with mass and moment of inertia, whose velocity and spin relax with friction.

    With unit friction, the position follows `m x'' = -x' + v0 n + sqrt(2 D_t) xi` along the orientation
    `n = (cos theta, sin theta)`, and the orientation follows `J theta'' = -theta' + sqrt(2 D_r) xi_r`. The velocity
    relaxes over the time `m` and the angular velocity over the time `J`, and both vanish to give the overdamped
    active Brownian particle.

    Args:
        label:
            The label of the particle or the group.

        visual:
            How the particle is drawn.

        x:
            The initial position.

        x_d1:
            The initial velocity.

        theta:
            The initial orientation.

        theta_d1:
            The initial angular velocity.

        speed:
            The self-propulsion speed `v0`.

        mass:
            The mass `m`, in units of the friction.

        moment:
            The moment of inertia `J`, in units of the rotational friction.

        D_t:
            The translational diffusion coefficient.

        D_r:
            The rotational diffusion coefficient.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import InertialActiveParticleElement
        >>> InertialActiveParticleElement('swimmer').mass
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = POINTS,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, order=2, role='position')] = 0.0,
        x_d1: npt.ArrayLike = 0.0,
        theta: Annotated[float, StateSpec(order=2, role='direction')] = 0.0,
        theta_d1: float = 0.0,
        speed: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        mass: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        moment: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        D_t: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        D_r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.theta = theta
        self.theta_d1 = theta_d1
        self.speed = speed
        self.mass = mass
        self.moment = moment
        self.D_t = D_t
        self.D_r = D_r

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the accelerations of the position and the orientation, and their noises.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        heading = vector(SR, [cos(s.theta), sin(s.theta)])
        return [
            RateLaw(s.x, (s.speed * heading - s.x_d1) / s.mass),
            NoiseLaw(s.x, sqrt(2 * s.D_t) / s.mass),
            RateLaw(s.theta, -s.theta_d1 / s.moment),
            NoiseLaw(s.theta, sqrt(2 * s.D_r) / s.moment),
        ]


class OrnsteinUhlenbeckParticleElement(Element):
    """Active Ornstein-Uhlenbeck particle, whose propulsion velocity `u` is an Ornstein-Uhlenbeck process.

    The laws are `dx = u dt + sqrt(2 D_t) dW` and `tau du = -u dt + sqrt(2 D) dW`, so the propulsion keeps its
    direction for the persistence time `tau` and the particle diffuses with `D + D_t` at long times.

    Args:
        label:
            The label of the particle or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.systems.physics.active_matter import OrnsteinUhlenbeckParticleElement
        >>> OrnsteinUhlenbeckParticleElement('swimmer').tau
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = POINTS,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
        u: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        D_t: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.u = u
        self.tau = tau
        self.D = D
        self.D_t = D_t

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the propulsion, its relaxation and noise, and the translational noise.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            RateLaw(s.x, s.u),
            NoiseLaw(s.x, sqrt(2 * s.D_t)),
            RateLaw(s.u, -s.u / s.tau),
            NoiseLaw(s.u, sqrt(2 * s.D) / s.tau),
        ]


class TumblerView(AgentView):
    """Run-and-tumble agent as rules see it, with its direction and the parameters of its group.

    Args:
        agent:
            The Mesa agent.

        stepper:
            The stepper that runs the system.
    """

    theta: float
    speed: float
    tumble_rate: float
    width: float


class TumblerElement(Element):
    """Run-and-tumble agent in a periodic square, which runs straight and tumbles to a random direction.

    Every step the agent moves `speed` along its direction, then tumbles with probability `1 - exp(-tumble_rate)`
    to a direction drawn uniformly, as bacteria such as E. coli do.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        theta:
            The initial direction.

        speed:
            The distance run per step.

        tumble_rate:
            The rate of tumbles per step.

        width:
            The side of the periodic square.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import TumblerElement
        >>> TumblerElement('bacterium').speed
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'theta'}),
        *,
        theta: Annotated[float, StateSpec(role='direction')] = 0.0,
        speed: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        tumble_rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        width: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.speed = speed
        self.tumble_rate = tumble_rate
        self.width = width

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the run-and-tumble rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.run_and_tumble)]

    @classmethod
    def _get_view(cls) -> type:
        """Return the class of the views of the element's agents."""
        return TumblerView

    def run_and_tumble(self, agent: TumblerView, system: SystemView) -> None:
        """Run one step along the direction, then tumble at random.

        Args:
            agent:
                The agent.

            system:
                The system.
        """
        position = np.asarray(agent.location, dtype=float)
        heading = np.array([math.cos(agent.theta), math.sin(agent.theta)])
        agent.move_to(np.mod(position + agent.speed * heading, agent.width))
        if system.rng.random() < -math.expm1(-agent.tumble_rate):
            agent.theta = float(system.rng.uniform(0.0, 2 * math.pi))


def _swimmers(
    n_particles: int,
    size: float,
    random_state: int | np.random.Generator | None,
) -> tuple[Array, Array]:
    """Return uniform positions in a square and uniform orientations."""
    rng = np.random.default_rng(random_state)
    return rng.uniform(0.0, size, (n_particles, 2)), rng.uniform(0.0, 2 * math.pi, n_particles)


@register_system(
    field='physics/active_matter',
    tags=('active matter', 'sde', 'self-propulsion', 'diffusion'),
    references=('Argun et al. 2021', 'Bechinger et al. 2016', 'Volpe, Gigan, and Volpe 2014'),
    defaults={'until': 20.0, 'dt': 0.5, 'solver': EulerMaruyamaSolver(dt=0.01)},
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
            The system, with the positions `swimmers.x` and orientations `swimmers.theta`.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import active_brownian_particles
        >>> active_brownian_particles(10).elements[0].prepare().size_
        10
    """
    positions, angles = _swimmers(n_particles, size, random_state)
    group = GroupElement(
        'swimmers',
        element=ActiveParticleElement,
        size=n_particles,
        values={'x': positions, 'theta': angles, 'D_t': D_t, 'D_r': D_r},
    )
    propulsion = SelfPropulsionInteraction('propulsion', target='swimmers', speed=speed)
    return System(elements=[group], interactions=[propulsion], space=ContinuousSpace(n_dim=2))


@register_system(
    field='physics/active_matter',
    tags=('active matter', 'sde', 'self-propulsion', 'chirality'),
    references=('Bechinger et al. 2016', 'van Teeffelen and Löwen 2008'),
    defaults={'until': 20.0, 'dt': 0.5, 'solver': EulerMaruyamaSolver(dt=0.01)},
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
            The system, with the positions `swimmers.x` and orientations `swimmers.theta`.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import chiral_active_particles
        >>> chiral_active_particles(10, omega=1.0).elements[0].values['omega']
        1.0
    """
    positions, angles = _swimmers(n_particles, size, random_state)
    group = GroupElement(
        'swimmers',
        element=ActiveParticleElement,
        size=n_particles,
        values={'x': positions, 'theta': angles, 'omega': omega, 'D_t': D_t, 'D_r': D_r},
    )
    propulsion = SelfPropulsionInteraction('propulsion', target='swimmers', speed=speed)
    return System(elements=[group], interactions=[propulsion], space=ContinuousSpace(n_dim=2))


@register_system(
    field='physics/active_matter',
    tags=('active matter', 'sde', 'self-propulsion', 'colored noise'),
    references=('Bechinger et al. 2016', 'Fodor et al. 2016'),
    defaults={'until': 20.0, 'dt': 0.5, 'solver': EulerMaruyamaSolver(dt=0.01)},
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
            The system, with the positions `swimmers.x` and propulsions `swimmers.u`.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import active_ornstein_uhlenbeck_particles
        >>> active_ornstein_uhlenbeck_particles(10, tau=0.5).elements[0].values['tau']
        0.5
    """
    rng = np.random.default_rng(random_state)
    positions = rng.uniform(0.0, size, (n_particles, 2))
    propulsion = rng.normal(0.0, math.sqrt(D / tau), (n_particles, 2))
    group = GroupElement(
        'swimmers',
        element=OrnsteinUhlenbeckParticleElement,
        size=n_particles,
        values={'x': positions, 'u': propulsion, 'tau': tau, 'D': D, 'D_t': D_t},
    )
    return System(elements=[group], space=ContinuousSpace(n_dim=2))


@register_system(
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
            The system, with the agent group `bacteria`.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import run_and_tumble
        >>> run_and_tumble(10).elements[0].prepare().size_
        10
    """
    positions, angles = _swimmers(n_agents, size, random_state)
    group = GroupElement(
        'bacteria',
        element=TumblerElement,
        size=n_agents,
        values={'theta': angles, 'speed': speed, 'tumble_rate': tumble_rate, 'width': size},
        location=positions,
    )
    return System(elements=[group], space=ContinuousSpace(extent=(size, size), boundary='periodic'))


@register_system(
    field='physics/active_matter',
    tags=('active matter', 'sde', 'energy depot', 'self-propulsion'),
    references=('Schweitzer, Ebeling, and Tilch 1998', 'Helbing 2001'),
    defaults={'until': 10.0, 'dt': 0.1, 'solver': EulerMaruyamaSolver(dt=0.001)},
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
            The system, with the positions `walkers.x`, velocities `walkers.v`, and depots `walkers.e`.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import depot_brownian_particles
        >>> depot_brownian_particles(5).elements[0].values['q']
        10.0
    """
    velocities = np.random.default_rng(random_state).normal(0.0, 0.01, (n_particles, 2))
    group = GroupElement(
        'walkers',
        element=DepotParticleElement,
        size=n_particles,
        values={'v': velocities, 'gamma': gamma, 'q': q, 'c': c, 'd': d, 'D': D},
    )
    return System(elements=[group], space=ContinuousSpace(n_dim=2))


@register_system(
    field='physics/active_matter',
    tags=('active matter', 'sde', 'self-propulsion', 'inertia', 'second order'),
    references=('Löwen 2020', 'Scholz et al. 2018', 'Bechinger et al. 2016'),
    defaults={'until': 20.0, 'dt': 0.5, 'solver': EulerMaruyamaSolver(dt=0.005)},
)
def inertial_active_brownian_particles(
    n_particles: int = 200,
    *,
    speed: float = 3.0,
    mass: float = 0.5,
    moment: float = 0.1,
    D_t: float = 0.1,
    D_r: float = 1.0,
    size: float = 50.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return active Brownian particles with inertia, which lag behind their orientation as they speed up and turn.

    With unit friction, every particle follows `m x'' = -x' + v0 n + sqrt(2 D_t) xi` and
    `J theta'' = -theta' + sqrt(2 D_r) xi_r`, the inertial active Brownian particles of macroscopic swimmers, flyers,
    and vibrated grains. The velocity relaxes over the time `m`, so it trails the orientation, and the mean projection
    of the velocity on the orientation falls below `v0`. With rotational inertia alone, the orientation correlation is
    `exp(-D_r (t - J (1 - exp(-t / J))))`, which gives the steady projection
    `v0 / m ∫ exp(-t / m) exp(-D_r (t - J (1 - exp(-t / J)))) dt`, and `v0 / (1 + m D_r)` without rotational inertia.
    The particles start at rest at uniform positions and orientations.

    Args:
        n_particles:
            The number of particles.

        speed:
            The self-propulsion speed `v0`.

        mass:
            The mass `m`, in units of the friction.

        moment:
            The moment of inertia `J`, in units of the rotational friction.

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
            The system, with the positions `swimmers.x`, the velocities `swimmers.x_d1`, the orientations
            `swimmers.theta`, and the angular velocities `swimmers.theta_d1`.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import inertial_active_brownian_particles
        >>> inertial_active_brownian_particles(10).elements[0].prepare().size_
        10
    """
    positions, angles = _swimmers(n_particles, size, random_state)
    group = GroupElement(
        'swimmers',
        element=InertialActiveParticleElement,
        size=n_particles,
        values={
            'x': positions,
            'theta': angles,
            'speed': speed,
            'mass': mass,
            'moment': moment,
            'D_t': D_t,
            'D_r': D_r,
        },
    )
    return System(elements=[group], space=ContinuousSpace(n_dim=2))
