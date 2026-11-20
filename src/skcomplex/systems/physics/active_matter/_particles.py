"""Active particle systems.

The module covers active Brownian, chiral, Ornstein-Uhlenbeck, run-and-tumble, and depot-pumped particles.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol

import numpy as np
from sage.all__sagemath_symbolics import SR, cos, oo, sin, sqrt, vector
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, Period
from ....core.components.interactions import Interaction
from ....core.components.laws import NoiseLaw, RateLaw, UpdateLaw
from ....core.components.spaces import BaseSpace, ContinuousSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, EulerMaruyamaSolver, SystemView
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ..common.particles import POINTS, ActiveDiffusionInteraction, ActiveParticleElement, SelfPropulsionInteraction


class DepotParticleElement(Element):
    """Particles with the positions `x`, the velocities `v`, and the energy depots `e`.

    Args:
        label:
            The label of the particles.

        x:
            The initial positions, one row per particle or shared.

        v:
            The initial velocities, one row per particle or shared.

        e:
            The initial energy of the depots.

        size:
            The number of particles, or `None` for one.

        space:
            The continuous space of the particles, or `None` for none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import DepotParticleElement
        >>> DepotParticleElement('walker', e=1.0).e
        1.0
    """

    visual = POINTS

    def __init__(
        self,
        label: str,
        *,
        x: object = (0.0, 0.0),
        v: object = (0.0, 0.0),
        e: object = 0.0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.x = x
        self.v = v
        self.e = e


class DepotPumpingInteraction(Interaction):
    """Brownian motion pumped by an internal energy depot.

    The particle follows the system of Schweitzer, Ebeling, and Tilch.

    The depot takes energy at the rate `q`, loses it at the rate `c e`, and converts it into motion at the rate
    `d e |v|^2`, so `de/dt = q - c e - d e |v|^2` and `dv = (-gamma v + d e v) dt + sqrt(2 D) dW`. Above
    `q d = c gamma` the particle moves at the speed `sqrt(q / gamma - c / d)`. It is a swimming cell that burns food, or
    a walker that spends stored energy.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particles, with the positions `x`, the velocities `v`, and the depots `e`.

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
        >>> from skcomplex.systems.physics.active_matter import DepotPumpingInteraction
        >>> DepotPumpingInteraction('pumping', target='walkers').q
        10.0
    """

    law_classes = (NoiseLaw, RateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        gamma: Annotated[float, RealSet.open(0, oo)] = 20.0,
        q: Annotated[float, RealSet.closed_open(0, oo)] = 10.0,
        c: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        d: Annotated[float, RealSet.closed_open(0, oo)] = 10.0,
        D: Annotated[float, RealSet.closed_open(0, oo)] = 0.01,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.gamma = gamma
        self.q = q
        self.c = c
        self.d = d
        self.D = D

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the motion, the pumped friction and the noise of the velocity, and the depot balance of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            x, v, e = s.target.x, s.target.v, s.target.e
            self.laws_ = [
                RateLaw(x, v),
                RateLaw(v, (-s.gamma + s.d * e) * v),
                NoiseLaw(v, sqrt(2 * s.D)),
                RateLaw(e.diff(s.t), s.q - s.c * e - s.d * e * v.dot_product(v)),
            ]


class InertialActiveParticleElement(Element):
    """Active particles with the positions `x` and the orientations `theta`, both of order 2.

    Args:
        label:
            The label of the particles.

        x:
            The initial positions, one row per particle or shared.

        theta:
            The initial orientations, one per particle or shared.

        size:
            The number of particles, or `None` for one.

        space:
            The continuous space of the particles, or `None` for none.

        values:
            Further values by name, such as the initial velocities `x'` and angular velocities `theta'`, or `None`.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import InertialActiveParticleElement
        >>> InertialActiveParticleElement('swimmer', theta=1.0).theta
        1.0
    """

    visual = POINTS

    def __init__(
        self,
        label: str,
        *,
        x: object = (0.0, 0.0),
        theta: object = 0.0,
        size: int | None = None,
        space: BaseSpace | None = None,
        values: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.x = x
        self.theta = theta


class InertialPropulsionInteraction(Interaction):
    """Self-propulsion of active particles with mass and moment of inertia.

    Their velocities and spins relax with friction.

    With unit friction, the position follows `m x'' = -x' + v0 n + sqrt(2 D_t) xi` along the orientation
    `n = (cos theta, sin theta)`, and the orientation follows `J theta'' = -theta' + sqrt(2 D_r) xi_r`. The velocity
    relaxes over the time `m` and the angular velocity over the time `J`, and both vanish to give the overdamped
    active Brownian particle. It is a vibrated grain, a flying insect, or a macroscopic robot.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particles, with the positions `x` and the orientations `theta`.

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
        >>> from skcomplex.systems.physics.active_matter import InertialPropulsionInteraction
        >>> InertialPropulsionInteraction('propulsion', target='swimmers').mass
        0.5
    """

    law_classes = (NoiseLaw, RateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        speed: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        mass: Annotated[float, RealSet.open(0, oo)] = 0.5,
        moment: Annotated[float, RealSet.open(0, oo)] = 0.1,
        D_t: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        D_r: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.speed = speed
        self.mass = mass
        self.moment = moment
        self.D_t = D_t
        self.D_r = D_r

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the accelerations of the position and the orientation of the target, and their noises.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context, context.t
            x, theta = s.target.x, s.target.theta
            heading = vector(SR, [cos(theta), sin(theta)])
            self.laws_ = [
                RateLaw(x.diff(t, 2), (s.speed * heading - x.diff(t)) / s.mass),
                NoiseLaw(x.diff(t, 2), sqrt(2 * s.D_t) / s.mass),
                RateLaw(theta.diff(t, 2), -theta.diff(t) / s.moment),
                NoiseLaw(theta.diff(t, 2), sqrt(2 * s.D_r) / s.moment),
            ]


class OrnsteinUhlenbeckParticleElement(Element):
    """Active particles with the positions `x` and the propulsion velocities `u`.

    Args:
        label:
            The label of the particles.

        x:
            The initial positions, one row per particle or shared.

        u:
            The initial propulsion velocities, one row per particle or shared.

        size:
            The number of particles, or `None` for one.

        space:
            The continuous space of the particles, or `None` for none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import OrnsteinUhlenbeckParticleElement
        >>> OrnsteinUhlenbeckParticleElement('swimmer', u=(1.0, 0.0)).u
        (1.0, 0.0)
    """

    visual = POINTS

    def __init__(
        self,
        label: str,
        *,
        x: object = (0.0, 0.0),
        u: object = (0.0, 0.0),
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.x = x
        self.u = u


class OrnsteinUhlenbeckPropulsionInteraction(Interaction):
    """Propulsion of active particles by an Ornstein-Uhlenbeck process.

    The laws are `dx = u dt + sqrt(2 D_t) dW` and `tau du = -u dt + sqrt(2 D) dW`, so the propulsion keeps its
    direction for the persistence time `tau` and the particle diffuses with `D + D_t` at long times. It is a bacterium
    or a colloid whose swimming direction forgets itself smoothly.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particles, with the positions `x` and the propulsions `u`.

        tau:
            The persistence time.

        D:
            The active diffusion coefficient.

        D_t:
            The translational diffusion coefficient.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import OrnsteinUhlenbeckPropulsionInteraction
        >>> OrnsteinUhlenbeckPropulsionInteraction('propulsion', target='swimmers').tau
        1.0
    """

    law_classes = (NoiseLaw, RateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        tau: Annotated[float, RealSet.open(0, oo)] = 1.0,
        D: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        D_t: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.tau = tau
        self.D = D
        self.D_t = D_t

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the propulsion of the position, the relaxation and noise of the propulsion, and the position noise.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            x, u = s.target.x, s.target.u
            self.laws_ = [
                RateLaw(x, u),
                NoiseLaw(x, sqrt(2 * s.D_t)),
                RateLaw(u, -u / s.tau),
                NoiseLaw(u, sqrt(2 * s.D) / s.tau),
            ]


class TumblerView(AgentViewProtocol, Protocol):
    """Run-and-tumble agent as rules see it.

    The agent has its direction and the parameters of its group.

    Args:
        agent:
            The Mesa agent.

        stepper:
            The stepper that runs the system.
    """

    x: float
    y: float
    theta: float
    speed: float
    tumble_rate: float
    width: float


class TumblerElement(Element):
    """Run-and-tumble agents at the points `x` and `y` of a periodic square, each with its direction `theta`.

    `RunAndTumbleInteraction` runs every agent straight and tumbles it to a random direction, as bacteria such as E.
    coli do. The direction wraps around the circle.

    Args:
        label:
            The label of the agent or group.

        x:
            The initial first coordinate, one per agent or shared.

        y:
            The initial second coordinate, one per agent or shared.

        theta:
            The initial direction, one per agent or shared.

        speed:
            The distance run per step.

        tumble_rate:
            The rate of tumbles per step.

        width:
            The side of the periodic square.

        size:
            The number of agents, or `None` for one.

        space:
            The continuous space of the agents, or `None` for none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import TumblerElement
        >>> TumblerElement('bacterium').speed
        0.1
    """

    visual = VisualSpec('agents', {'color_by': 'theta'})

    def __init__(
        self,
        label: str,
        *,
        x: object = 0.0,
        y: object = 0.0,
        theta: Annotated[object, Period(2 * math.pi)] = 0.0,
        speed: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        tumble_rate: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        width: Annotated[float, RealSet.open(0, oo)] = 10.0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.x = x
        self.y = y
        self.theta = theta
        self.speed = speed
        self.tumble_rate = tumble_rate
        self.width = width


class RunAndTumbleInteraction(Interaction):
    """Run of every agent along its direction, then a tumble to a random direction.

    Every step the agent moves `speed` along its direction in a periodic square, then tumbles with probability
    `1 - exp(-tumble_rate)` to a direction drawn uniformly, as bacteria swim and robots search.

    Args:
        label:
            The label of the interaction.

        target:
            The agents, with the points `x` and `y`, the direction `theta`, and the parameters of the run.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import RunAndTumbleInteraction
        >>> RunAndTumbleInteraction('run', target='bacteria').target
        'bacteria'
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the run and tumble of every agent as the rule.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context.target, self._run_and_tumble)]

    def _run_and_tumble(self, agent: TumblerView, system: SystemView) -> None:
        """Run one step along the direction, then tumble at random.

        Args:
            agent:
                The agent.

            system:
                The system.
        """
        agent.x = float(np.mod(agent.x + agent.speed * math.cos(agent.theta), agent.width))
        agent.y = float(np.mod(agent.y + agent.speed * math.sin(agent.theta), agent.width))
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ActiveBrownianParticlesSystem
        >>> ActiveBrownianParticlesSystem(10).prepare().system_.elements[0].size
        10
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'sde', 'self-propulsion', 'diffusion')
    references = ('Argun et al. 2021', 'Bechinger et al. 2016', 'Volpe, Gigan, and Volpe 2014')
    until = 20.0
    dt = 0.5

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.speed = speed
        self.D_t = D_t
        self.D_r = D_r
        self.size = size
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of active Brownian particles in the plane.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        positions, angles = ActiveBrownianParticlesSystem._get_swimmers(self.n_particles, self.size, self.random_state)
        swimmers = ActiveParticleElement(
            'swimmers',
            x=positions,
            theta=angles,
            size=self.n_particles,
            space=ContinuousSpace(n_dim=2),
        )
        self.system_ = System(
            self.label,
            elements=[swimmers],
            interactions=[
                ActiveDiffusionInteraction('diffusion', target='swimmers', D_t=self.D_t, D_r=self.D_r),
                SelfPropulsionInteraction('propulsion', target='swimmers', speed=self.speed),
            ],
            solver=EulerMaruyamaSolver(dt=0.01),
        )


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ChiralActiveParticlesSystem
        >>> ChiralActiveParticlesSystem(10, omega=1.0).prepare().system_.interactions[0].omega
        1.0
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'sde', 'self-propulsion', 'chirality')
    references = ('Bechinger et al. 2016', 'van Teeffelen and Löwen 2008')
    until = 20.0
    dt = 0.5

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.speed = speed
        self.omega = omega
        self.D_t = D_t
        self.D_r = D_r
        self.size = size
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of chiral active particles.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        positions, angles = ActiveBrownianParticlesSystem._get_swimmers(self.n_particles, self.size, self.random_state)
        swimmers = ActiveParticleElement(
            'swimmers',
            x=positions,
            theta=angles,
            size=self.n_particles,
            space=ContinuousSpace(n_dim=2),
        )
        diffusion = ActiveDiffusionInteraction(
            'diffusion',
            target='swimmers',
            D_t=self.D_t,
            D_r=self.D_r,
            omega=self.omega,
        )
        self.system_ = System(
            self.label,
            elements=[swimmers],
            interactions=[diffusion, SelfPropulsionInteraction('propulsion', target='swimmers', speed=self.speed)],
            solver=EulerMaruyamaSolver(dt=0.01),
        )


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ActiveOrnsteinUhlenbeckParticlesSystem
        >>> ActiveOrnsteinUhlenbeckParticlesSystem(10, tau=0.5).prepare().system_.interactions[0].tau
        0.5
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'sde', 'self-propulsion', 'colored noise')
    references = ('Bechinger et al. 2016', 'Fodor et al. 2016')
    until = 20.0
    dt = 0.5

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.tau = tau
        self.D = D
        self.D_t = D_t
        self.size = size
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of active Ornstein-Uhlenbeck particles in the plane.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        positions = rng.uniform(0.0, self.size, (self.n_particles, 2))
        propulsion = rng.normal(0.0, math.sqrt(self.D / self.tau), (self.n_particles, 2))
        swimmers = OrnsteinUhlenbeckParticleElement(
            'swimmers',
            x=positions,
            u=propulsion,
            size=self.n_particles,
            space=ContinuousSpace(n_dim=2),
        )
        self.system_ = System(
            self.label,
            elements=[swimmers],
            interactions=[
                OrnsteinUhlenbeckPropulsionInteraction(
                    'propulsion',
                    target='swimmers',
                    tau=self.tau,
                    D=self.D,
                    D_t=self.D_t,
                ),
            ],
            solver=EulerMaruyamaSolver(dt=0.01),
        )


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import RunAndTumbleSystem
        >>> RunAndTumbleSystem(10).prepare().system_.elements[0].size
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_agents = n_agents
        self.speed = speed
        self.tumble_rate = tumble_rate
        self.size = size
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of run-and-tumble agents in a periodic square.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        positions, angles = ActiveBrownianParticlesSystem._get_swimmers(self.n_agents, self.size, self.random_state)
        bacteria = TumblerElement(
            'bacteria',
            x=positions[:, 0],
            y=positions[:, 1],
            theta=angles,
            speed=self.speed,
            tumble_rate=self.tumble_rate,
            width=self.size,
            size=self.n_agents,
            space=ContinuousSpace(extent=(self.size, self.size), boundary='periodic'),
        )
        run = RunAndTumbleInteraction('run', target='bacteria')
        self.system_ = System(self.label, elements=[bacteria], interactions=[run])


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import DepotBrownianParticlesSystem
        >>> DepotBrownianParticlesSystem(5).prepare().system_.interactions[0].q
        10.0
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'sde', 'energy depot', 'self-propulsion')
    references = ('Schweitzer, Ebeling, and Tilch 1998', 'Helbing 2001')
    until = 10.0
    dt = 0.1

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.gamma = gamma
        self.q = q
        self.c = c
        self.d = d
        self.D = D
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of Brownian particles with energy depots.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        velocities = np.random.default_rng(self.random_state).normal(0.0, 0.01, (self.n_particles, 2))
        walkers = DepotParticleElement(
            'walkers',
            x=np.zeros((self.n_particles, 2)),
            v=velocities,
            size=self.n_particles,
            space=ContinuousSpace(n_dim=2),
        )
        pumping = DepotPumpingInteraction(
            'pumping',
            target='walkers',
            gamma=self.gamma,
            q=self.q,
            c=self.c,
            d=self.d,
            D=self.D,
        )
        self.system_ = System(
            self.label,
            elements=[walkers],
            interactions=[pumping],
            solver=EulerMaruyamaSolver(dt=0.001),
        )


class InertialActiveBrownianParticlesSystem(BaseSystem):
    """Active Brownian particles with inertia.

    The particles lag behind their orientation as they speed up and turn.

    With unit friction, every particle follows `m x'' = -x' + v0 n + sqrt(2 D_t) xi` and `J theta'' = -theta' + sqrt(2
    D_r) xi_r`, the inertial active Brownian particles of macroscopic swimmers, flyers, and vibrated grains. The
    velocity relaxes over the time `m`, so it trails the orientation, and the mean projection of the velocity on the
    orientation falls below `v0`. With rotational inertia alone, the orientation correlation is `exp(-D_r (t - J (1 -
    exp(-t / J))))`, which gives the steady projection `v0 / m ∫ exp(-t / m) exp(-D_r (t - J (1 - exp(-t / J)))) dt`,
    and `v0 / (1 + m D_r)` without rotational inertia. The particles start at rest at uniform positions and
    orientations. The system has the positions `swimmers.x`, the velocities `swimmers.x'`, the orientations
    `swimmers.theta`, and the angular velocities `swimmers.theta'`.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import InertialActiveBrownianParticlesSystem
        >>> InertialActiveBrownianParticlesSystem(10).prepare().system_.elements[0].size
        10
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'sde', 'self-propulsion', 'inertia', 'second order')
    references = ('Löwen 2020', 'Scholz et al. 2018', 'Bechinger et al. 2016')
    until = 20.0
    dt = 0.5

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.speed = speed
        self.mass = mass
        self.moment = moment
        self.D_t = D_t
        self.D_r = D_r
        self.size = size
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of active Brownian particles with inertia.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        positions, angles = ActiveBrownianParticlesSystem._get_swimmers(self.n_particles, self.size, self.random_state)
        swimmers = InertialActiveParticleElement(
            'swimmers',
            x=positions,
            theta=angles,
            size=self.n_particles,
            space=ContinuousSpace(n_dim=2),
        )
        propulsion = InertialPropulsionInteraction(
            'propulsion',
            target='swimmers',
            speed=self.speed,
            mass=self.mass,
            moment=self.moment,
            D_t=self.D_t,
            D_r=self.D_r,
        )
        self.system_ = System(
            self.label,
            elements=[swimmers],
            interactions=[propulsion],
            solver=EulerMaruyamaSolver(dt=0.005),
        )
