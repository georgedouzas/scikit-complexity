"""Collective active matter models: motility-induced phase separation, living crystals, flocking, and Toner-Tu."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence

import numpy as np

from ....core.elements import FieldElement, GroupElement, PointElement
from ....core.spaces import ContinuousSpace, LatticeSpace
from ....core.system import System
from ....engine.solvers import EulerMaruyamaSolver, RK4Solver
from ....specs import VisualSpec
from ....types import Array
from ..._catalog import register_model
from ..common.particles import SoftSphereInteraction
from ._active_particle import ActiveParticleElement
from ._cucker_smale_alignment import CuckerSmaleAlignmentInteraction
from ._phoretic_attraction import PhoreticAttractionInteraction
from ._polar_fluid import PolarFluidElement
from ._self_propulsion import SelfPropulsionInteraction


def _disks(
    n_particles: int,
    size: float,
    random_state: int | np.random.Generator | None,
) -> tuple[Array, Array]:
    """Return positions on a jittered square grid that fills a box, and uniform orientations."""
    rng = np.random.default_rng(random_state)
    side = math.ceil(math.sqrt(n_particles))
    spacing = size / side
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2)[:n_particles]
    positions = np.mod((grid + 0.5) * spacing + rng.normal(0.0, 0.05 * spacing, (n_particles, 2)), size)
    return positions, rng.uniform(0.0, 2 * math.pi, n_particles)


def _active_disks(
    n_particles: int,
    size: float,
    speed: float,
    D_r: float,
    random_state: int | np.random.Generator | None,
) -> System:
    """Return active Brownian disks in a periodic box, without pair interactions."""
    positions, angles = _disks(n_particles, size, random_state)
    group = GroupElement(
        'swimmers',
        element=ActiveParticleElement,
        size=n_particles,
        values={'x': positions, 'theta': angles, 'D_t': 0.0, 'D_r': D_r},
    )
    propulsion = SelfPropulsionInteraction('propulsion', target='swimmers', speed=speed)
    return System(
        elements=[group],
        interactions=[propulsion],
        space=ContinuousSpace(extent=(size, size), boundary='periodic'),
    )


@register_model(
    field='physics/active_matter',
    tags=('active matter', 'sde', 'phase separation', 'pair interaction'),
    references=('Cates and Tailleur 2015', 'Bechinger et al. 2016', 'Fily and Marchetti 2012'),
    defaults={'until': 20.0, 'dt': 0.1, 'solver': EulerMaruyamaSolver(dt=0.005)},
)
def motility_induced_phase_separation(
    n_particles: int = 400,
    *,
    packing_fraction: float = 0.5,
    speed: float = 1.0,
    D_r: float = 0.01,
    k: float = 50.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return repulsive active Brownian disks of unit diameter, which phase separate into clusters at high speed.

    Disks that collide head on block each other until their orientations diffuse away, so fast swimmers with slow
    rotational diffusion, a large Péclet number `v / D_r`, accumulate where they are slow and form dense clusters in a
    dilute gas.

    Args:
        n_particles:
            The number of disks.

        packing_fraction:
            The fraction of the periodic box the disks cover.

        speed:
            The self-propulsion speed.

        D_r:
            The rotational diffusion coefficient.

        k:
            The stiffness of the repulsion times the mobility.

        random_state:
            The seed or generator of the initial positions and orientations.

    Returns:
        system:
            The model, with the positions `swimmers.x` and orientations `swimmers.theta`.

    Examples:
        >>> from skcomplex.models.physics.active_matter import motility_induced_phase_separation
        >>> motility_induced_phase_separation(16).interactions[1].k
        50.0
    """
    size = math.sqrt(n_particles * math.pi / 4 / packing_fraction)
    system = _active_disks(n_particles, size, speed, D_r, random_state)
    repulsion = SoftSphereInteraction('contact', first='swimmers', second='swimmers', cutoff=1.0, period=size, k=k)
    return System(elements=system.elements, interactions=[*system.interactions, repulsion], space=system.space)


@register_model(
    field='physics/active_matter',
    tags=('active matter', 'sde', 'clusters', 'phoresis', 'pair interaction'),
    references=('Argun et al. 2021', 'Palacci et al. 2013'),
    defaults={'until': 20.0, 'dt': 0.1, 'solver': EulerMaruyamaSolver(dt=0.002)},
)
def living_crystals(
    n_particles: int = 100,
    *,
    size: float = 50.0,
    speed: float = 3.0,
    D_r: float = 1.0,
    v0: float = 20.0,
    cutoff: float = 10.0,
    k: float = 200.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return active colloids of radius 1 with phoretic attraction, which gather into clusters that slowly coarsen.

    The colloids swim as active Brownian particles, drift towards each other at the speed `v0 / r^2` within the
    cutoff, and a stiff soft repulsion stands in for the hard-sphere correction at contact.

    Args:
        n_particles:
            The number of colloids.

        size:
            The side of the periodic box.

        speed:
            The self-propulsion speed.

        D_r:
            The rotational diffusion coefficient.

        v0:
            The phoretic drift speed at the distance of one radius.

        cutoff:
            The distance beyond which the attraction is neglected.

        k:
            The stiffness of the repulsion times the mobility.

        random_state:
            The seed or generator of the initial positions and orientations.

    Returns:
        system:
            The model, with the positions `swimmers.x` and orientations `swimmers.theta`.

    Examples:
        >>> from skcomplex.models.physics.active_matter import living_crystals
        >>> living_crystals(16).interactions[2].v0
        20.0
    """
    system = _active_disks(n_particles, size, speed, D_r, random_state)
    repulsion = SoftSphereInteraction('contact', first='swimmers', second='swimmers', cutoff=2.0, period=size, k=k)
    attraction = PhoreticAttractionInteraction(
        'phoresis',
        first='swimmers',
        second='swimmers',
        cutoff=cutoff,
        period=size,
        v0=v0,
        R=1.0,
    )
    interactions = [*system.interactions, repulsion, attraction]
    return System(elements=system.elements, interactions=interactions, space=system.space)


@register_model(
    field='physics/active_matter',
    tags=('flocking', 'ode', 'alignment', 'pair interaction'),
    references=('Cucker and Smale 2007', 'Vicsek and Zafeiris 2012', 'Bellomo et al. 2017'),
    defaults={'until': 20.0, 'dt': 0.5, 'solver': RK4Solver(dt=0.05)},
)
def cucker_smale_flocking(
    n_agents: int = 50,
    *,
    K: float = 1.0,
    beta: float = 0.25,
    size: float = 10.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Cucker-Smale flocking, where every agent aligns its velocity with all others, weighted by distance.

    The mean velocity is conserved, and for `beta <= 1/2` the velocities converge to it from any start.

    Args:
        n_agents:
            The number of agents.

        K:
            The coupling strength.

        beta:
            The decay exponent of the communication weight.

        size:
            The side of the square in which the agents start.

        random_state:
            The seed or generator of the initial positions and velocities.

    Returns:
        system:
            The model, with the positions `birds.x` and velocities `birds.x_d1`.

    Examples:
        >>> from skcomplex.models.physics.active_matter import cucker_smale_flocking
        >>> cucker_smale_flocking(5).interactions[0].n_agents
        5
    """
    rng = np.random.default_rng(random_state)
    birds = GroupElement(
        'birds',
        element=PointElement,
        size=n_agents,
        values={'x': rng.uniform(0.0, size, (n_agents, 2)), 'x_d1': rng.normal(0.0, 1.0, (n_agents, 2))},
        visual=VisualSpec('points', {'position': 'x', 'color_by': 'x_d1', 'states': ()}),
    )
    alignment = CuckerSmaleAlignmentInteraction(
        'align',
        first='birds',
        second='birds',
        cutoff=1e6,
        K=K,
        beta=beta,
        n_agents=n_agents,
    )
    return System(elements=[birds], interactions=[alignment], space=ContinuousSpace(n_dim=2))


@register_model(
    field='physics/active_matter',
    tags=('field', 'flocking', 'hydrodynamics', 'pde'),
    references=('Toner and Tu 1995', 'Marchetti et al. 2013'),
    defaults={'until': 20.0, 'dt': 0.5, 'solver': EulerMaruyamaSolver(dt=0.02)},
)
def toner_tu(
    shape: Sequence[int] = (32, 32),
    *,
    rho0: float = 1.0,
    rho_c: float = 0.5,
    a: float = 1.0,
    beta: float = 1.0,
    lam: float = 1.0,
    sigma: float = 0.5,
    D: float = 0.5,
    spacing: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Toner-Tu hydrodynamics of a polar flock on a periodic lattice, from a disordered velocity field.

    Above the critical density the velocity grows and aligns until the whole fluid moves together with the speed
    `sqrt(a (rho0 - rho_c) / beta)`.

    Args:
        shape:
            The number of sites along each axis.

        rho0:
            The mean density.

        rho_c:
            The critical density of the ordering transition.

        a:
            The rate of the ordering instability per unit density.

        beta:
            The saturation of the speed.

        lam:
            The strength of the self-advection.

        sigma:
            The pressure per unit density.

        D:
            The viscosity of the velocity.

        spacing:
            The distance between sites.

        random_state:
            The seed or generator of the initial velocities.

    Returns:
        system:
            The model, with the fields `flock.rho`, `flock.vx`, and `flock.vy`.

    Examples:
        >>> from skcomplex.models.physics.active_matter import toner_tu
        >>> toner_tu((8, 8)).elements[0].prepare().size_
        64
    """
    shape = tuple(shape)
    rng = np.random.default_rng(random_state)
    values = {
        'rho': np.full(shape, rho0),
        'vx': rng.normal(0.0, 0.1, shape),
        'vy': rng.normal(0.0, 0.1, shape) + 0.05,
    }
    parameters = {'a': a, 'rho_c': rho_c, 'beta': beta, 'lam': lam, 'sigma': sigma, 'D': D}
    space = LatticeSpace(shape, spacing=spacing)
    return System(
        elements=[FieldElement('flock', element=PolarFluidElement, space=space, values={**values, **parameters})],
    )
