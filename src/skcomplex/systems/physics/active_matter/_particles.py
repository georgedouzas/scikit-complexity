"""Active particle systems: active Brownian, chiral, Ornstein-Uhlenbeck, run-and-tumble, and depot-pumped particles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.spaces import ContinuousSpace
from ....core.components.system import System
from ....core.engine.solvers import EulerMaruyamaSolver
from ....core.types import Array
from ..._catalog import register_system
from ._active_particle import ActiveParticleElement
from ._depot_particle import DepotParticleElement
from ._inertial_particle import InertialActiveParticleElement
from ._ornstein_uhlenbeck_particle import OrnsteinUhlenbeckParticleElement
from ._self_propulsion import SelfPropulsionInteraction
from ._tumbler import TumblerElement


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
