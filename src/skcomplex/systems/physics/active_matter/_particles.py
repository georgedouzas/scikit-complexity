"""Active particle systems.

The module covers active Brownian, chiral, Ornstein-Uhlenbeck, run-and-tumble, and depot-pumped particles.
"""

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
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import AgentView, EulerMaruyamaSolver, SystemView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..common.particles import POINTS, ActiveParticleElement, SelfPropulsionInteraction


class DepotParticleElement(Element):
    """Brownian particle pumped by an internal energy depot.

    The particle follows the system of Schweitzer, Ebeling, and Tilch.

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

    law_classes = (
        NoiseLaw,
        RateLaw,
    )

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
        """Return the laws of the particle.

        They are the motion, the friction and pumping of the velocity, its noise, and the depot balance.

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
    """Active Brownian particle with mass and moment of inertia.

    Its velocity and spin relax with friction.

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

    law_classes = (
        NoiseLaw,
        RateLaw,
    )

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
        """Return the accelerations of the position and orientation, and their noises.

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
    """Active Ornstein-Uhlenbeck particle.

    Its propulsion velocity `u` is an Ornstein-Uhlenbeck process.

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

    law_classes = (
        NoiseLaw,
        RateLaw,
    )

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
    """Run-and-tumble agent as rules see it.

    The agent has its direction and the parameters of its group.

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
    """Run-and-tumble agent in a periodic square.

    The agent runs straight and tumbles to a random direction.

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

    draws = True

    law_classes = (RuleLaw,)

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


class ActiveBrownianParticlesSystem(BaseSystem):
    """Active Brownian particles in the plane.

    The particles swim at a constant speed in a diffusing direction.

    Their mean squared displacement is ballistic below the persistence time `1 / D_r` and diffusive above it, with the
    effective diffusion coefficient `D_t + v^2 / (2 D_r)`. The system has the positions `swimmers.x` and orientations
    `swimmers.theta`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ActiveBrownianParticlesSystem
        >>> ActiveBrownianParticlesSystem(10).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'sde', 'self-propulsion', 'diffusion')
    references = ('Argun et al. 2021', 'Bechinger et al. 2016', 'Volpe, Gigan, and Volpe 2014')
    until = 20.0
    dt = 0.5
    solver = EulerMaruyamaSolver(dt=0.01)

    @staticmethod
    def _get_swimmers(
        n_particles: int,
        size: float,
        random_state: int | np.random.Generator | None,
    ) -> tuple[Array, Array]:
        """Return uniform positions in a square and uniform orientations."""
        rng = np.random.default_rng(random_state)
        return rng.uniform(0.0, size, (n_particles, 2)), rng.uniform(0.0, 2 * math.pi, n_particles)

    def __init__(
        self,
        n_particles: int = 200,
        *,
        speed: float = 3.0,
        D_t: float = 0.1,
        D_r: float = 1.0,
        size: float = 50.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_particles = n_particles
        self.speed = speed
        self.D_t = D_t
        self.D_r = D_r
        self.size = size
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return active Brownian particles in the plane.

        The particles swim at a constant speed in a diffusing direction.
        """
        positions, angles = ActiveBrownianParticlesSystem._get_swimmers(self.n_particles, self.size, self.random_state)
        group = GroupElement(
            'swimmers',
            element=ActiveParticleElement,
            size=self.n_particles,
            values={'x': positions, 'theta': angles, 'D_t': self.D_t, 'D_r': self.D_r},
        )
        propulsion = SelfPropulsionInteraction('propulsion', target='swimmers', speed=self.speed)
        return CompositeSystem(self.label, elements=[group], interactions=[propulsion], space=ContinuousSpace(n_dim=2))


class ChiralActiveParticlesSystem(BaseSystem):
    """Chiral active particles.

    The particles swim along circles of radius `v / omega` that drift by diffusion.

    Their effective diffusion coefficient at long times is `D_t + v^2 D_r / (2 (D_r^2 + omega^2))`. The system has the
    positions `swimmers.x` and orientations `swimmers.theta`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ChiralActiveParticlesSystem
        >>> ChiralActiveParticlesSystem(10, omega=1.0).prepare().system_.elements[0].values['omega']
        1.0
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'sde', 'self-propulsion', 'chirality')
    references = ('Bechinger et al. 2016', 'van Teeffelen and Löwen 2008')
    until = 20.0
    dt = 0.5
    solver = EulerMaruyamaSolver(dt=0.01)

    def __init__(
        self,
        n_particles: int = 200,
        *,
        speed: float = 3.0,
        omega: float = 2.0,
        D_t: float = 0.1,
        D_r: float = 0.1,
        size: float = 50.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_particles = n_particles
        self.speed = speed
        self.omega = omega
        self.D_t = D_t
        self.D_r = D_r
        self.size = size
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return chiral active particles.

        The particles swim along circles of radius `v / omega` that drift by diffusion.
        """
        positions, angles = ActiveBrownianParticlesSystem._get_swimmers(self.n_particles, self.size, self.random_state)
        group = GroupElement(
            'swimmers',
            element=ActiveParticleElement,
            size=self.n_particles,
            values={'x': positions, 'theta': angles, 'omega': self.omega, 'D_t': self.D_t, 'D_r': self.D_r},
        )
        propulsion = SelfPropulsionInteraction('propulsion', target='swimmers', speed=self.speed)
        return CompositeSystem(self.label, elements=[group], interactions=[propulsion], space=ContinuousSpace(n_dim=2))


class ActiveOrnsteinUhlenbeckParticlesSystem(BaseSystem):
    """Active Ornstein-Uhlenbeck particles in the plane.

    The particles are driven by exponentially correlated propulsion.

    The propulsion starts in its stationary distribution, normal with variance `D / tau` per component. The system has
    the positions `swimmers.x` and propulsions `swimmers.u`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ActiveOrnsteinUhlenbeckParticlesSystem
        >>> ActiveOrnsteinUhlenbeckParticlesSystem(10, tau=0.5).prepare().system_.elements[0].values['tau']
        0.5
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'sde', 'self-propulsion', 'colored noise')
    references = ('Bechinger et al. 2016', 'Fodor et al. 2016')
    until = 20.0
    dt = 0.5
    solver = EulerMaruyamaSolver(dt=0.01)

    def __init__(
        self,
        n_particles: int = 200,
        *,
        tau: float = 1.0,
        D: float = 1.0,
        D_t: float = 0.0,
        size: float = 50.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_particles = n_particles
        self.tau = tau
        self.D = D
        self.D_t = D_t
        self.size = size
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return active Ornstein-Uhlenbeck particles in the plane.

        The particles are driven by exponentially correlated propulsion.
        """
        rng = np.random.default_rng(self.random_state)
        positions = rng.uniform(0.0, self.size, (self.n_particles, 2))
        propulsion = rng.normal(0.0, math.sqrt(self.D / self.tau), (self.n_particles, 2))
        group = GroupElement(
            'swimmers',
            element=OrnsteinUhlenbeckParticleElement,
            size=self.n_particles,
            values={'x': positions, 'u': propulsion, 'tau': self.tau, 'D': self.D, 'D_t': self.D_t},
        )
        return CompositeSystem(self.label, elements=[group], space=ContinuousSpace(n_dim=2))


class RunAndTumbleSystem(BaseSystem):
    """Run-and-tumble agents in a periodic square.

    The agents diffuse with `v^2 (1 + q) / (4 (1 - q))` per step.

    The persistence `q = exp(-tumble_rate)` is the probability that an agent keeps its direction for one step. The
    system has the agent group `bacteria`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import RunAndTumbleSystem
        >>> RunAndTumbleSystem(10).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'agents', 'bacteria', 'self-propulsion')
    references = ('Berg 2004', 'Bechinger et al. 2016')
    until = 100
    dt = 1

    def __init__(
        self,
        n_agents: int = 100,
        *,
        speed: float = 0.1,
        tumble_rate: float = 0.1,
        size: float = 10.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_agents = n_agents
        self.speed = speed
        self.tumble_rate = tumble_rate
        self.size = size
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        positions, angles = ActiveBrownianParticlesSystem._get_swimmers(self.n_agents, self.size, self.random_state)
        group = GroupElement(
            'bacteria',
            element=TumblerElement,
            size=self.n_agents,
            values={'theta': angles, 'speed': self.speed, 'tumble_rate': self.tumble_rate, 'width': self.size},
            location=positions,
        )
        return CompositeSystem(
            self.label,
            elements=[group],
            space=ContinuousSpace(extent=(self.size, self.size), boundary='periodic'),
        )


class DepotBrownianParticlesSystem(BaseSystem):
    """Brownian particles with energy depots.

    The particles start at rest and take up the speed `sqrt(q/gamma - c/d)`.

    The system has the positions `walkers.x`, velocities `walkers.v`, and depots `walkers.e`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import DepotBrownianParticlesSystem
        >>> DepotBrownianParticlesSystem(5).prepare().system_.elements[0].values['q']
        10.0
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'sde', 'energy depot', 'self-propulsion')
    references = ('Schweitzer, Ebeling, and Tilch 1998', 'Helbing 2001')
    until = 10.0
    dt = 0.1
    solver = EulerMaruyamaSolver(dt=0.001)

    def __init__(
        self,
        n_particles: int = 50,
        *,
        gamma: float = 20.0,
        q: float = 10.0,
        c: float = 1.0,
        d: float = 10.0,
        D: float = 0.01,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_particles = n_particles
        self.gamma = gamma
        self.q = q
        self.c = c
        self.d = d
        self.D = D
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        velocities = np.random.default_rng(self.random_state).normal(0.0, 0.01, (self.n_particles, 2))
        group = GroupElement(
            'walkers',
            element=DepotParticleElement,
            size=self.n_particles,
            values={'v': velocities, 'gamma': self.gamma, 'q': self.q, 'c': self.c, 'd': self.d, 'D': self.D},
        )
        return CompositeSystem(self.label, elements=[group], space=ContinuousSpace(n_dim=2))


class InertialActiveBrownianParticlesSystem(BaseSystem):
    """Active Brownian particles with inertia.

    The particles lag behind their orientation as they speed up and turn.

    With unit friction, every particle follows `m x'' = -x' + v0 n + sqrt(2 D_t) xi` and `J theta'' = -theta' + sqrt(2
    D_r) xi_r`, the inertial active Brownian particles of macroscopic swimmers, flyers, and vibrated grains. The
    velocity relaxes over the time `m`, so it trails the orientation, and the mean projection of the velocity on the
    orientation falls below `v0`. With rotational inertia alone, the orientation correlation is `exp(-D_r (t - J (1 -
    exp(-t / J))))`, which gives the steady projection `v0 / m ∫ exp(-t / m) exp(-D_r (t - J (1 - exp(-t / J)))) dt`,
    and `v0 / (1 + m D_r)` without rotational inertia. The particles start at rest at uniform positions and
    orientations. The system has the positions `swimmers.x`, the velocities `swimmers.x_d1`, the orientations
    `swimmers.theta`, and the angular velocities `swimmers.theta_d1`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import InertialActiveBrownianParticlesSystem
        >>> InertialActiveBrownianParticlesSystem(10).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'sde', 'self-propulsion', 'inertia', 'second order')
    references = ('Löwen 2020', 'Scholz et al. 2018', 'Bechinger et al. 2016')
    until = 20.0
    dt = 0.5
    solver = EulerMaruyamaSolver(dt=0.005)

    def __init__(
        self,
        n_particles: int = 200,
        *,
        speed: float = 3.0,
        mass: float = 0.5,
        moment: float = 0.1,
        D_t: float = 0.1,
        D_r: float = 1.0,
        size: float = 50.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_particles = n_particles
        self.speed = speed
        self.mass = mass
        self.moment = moment
        self.D_t = D_t
        self.D_r = D_r
        self.size = size
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        positions, angles = ActiveBrownianParticlesSystem._get_swimmers(self.n_particles, self.size, self.random_state)
        group = GroupElement(
            'swimmers',
            element=InertialActiveParticleElement,
            size=self.n_particles,
            values={
                'x': positions,
                'theta': angles,
                'speed': self.speed,
                'mass': self.mass,
                'moment': self.moment,
                'D_t': self.D_t,
                'D_r': self.D_r,
            },
        )
        return CompositeSystem(self.label, elements=[group], space=ContinuousSpace(n_dim=2))
