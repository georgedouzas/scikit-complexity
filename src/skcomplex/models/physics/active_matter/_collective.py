"""Collective active matter models: motility-induced phase separation, living crystals, flocking, and Toner-Tu."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import sqrt

from ....core.base import Domain, VisualSpec, parameter, state
from ....core.model import (
    Element,
    EuclideanSpace,
    Field,
    LatticeSpace,
    Law,
    Noise,
    PairInteraction,
    Particle,
    PeriodicBox,
    Rate,
    Symbols,
    System,
    advection,
    divergence,
    gradient,
    laplacian,
)
from ....core.runtime import register_model
from ....core.solvers import RK4, EulerMaruyama
from ._particles import ActiveParticle

type Array = npt.NDArray[np.float64]


class SoftRepulsion(PairInteraction):
    """Overdamped repulsion between overlapping disks, which pushes them apart at the speed `k (cutoff - r)`.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the group or particle of the first members.

        second:
            The path of the group or particle of the second members.

        cutoff:
            The contact distance, the sum of the two radii.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

        k:
            The stiffness times the mobility.

    Examples:
        >>> from skcomplex.models.physics.active_matter import SoftRepulsion
        >>> SoftRepulsion('contact', first='swimmers', second='swimmers', k=50.0).k
        50.0
    """

    k: float = parameter(default=100.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the opposite displacements of the two members of an overlapping pair.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        separation = s.first.x - s.second.x
        r = sqrt(separation.dot_product(separation))
        push = s.k * (s.cutoff - r) * separation / r
        return [Rate(s.first.x, push), Rate(s.second.x, -push)]


class PhoreticAttraction(PairInteraction):
    """Phoretic attraction between active colloids, a drift towards each other at the speed `v0 R^2 / r^2`.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the group or particle of the first members.

        second:
            The path of the group or particle of the second members.

        cutoff:
            The distance beyond which the attraction is neglected.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

        v0:
            The drift speed at the distance `R`.

        R:
            The particle radius.

    Examples:
        >>> from skcomplex.models.physics.active_matter import PhoreticAttraction
        >>> PhoreticAttraction('phoresis', first='colloids', second='colloids', v0=20.0).v0
        20.0
    """

    v0: float = parameter(default=20.0, domain=Domain.NONNEGATIVE)
    R: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the drifts of the two members of a pair towards each other.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        separation = s.first.x - s.second.x
        r = sqrt(separation.dot_product(separation))
        drift = s.v0 * s.R**2 / r**2 * separation / r
        return [Rate(s.first.x, -drift), Rate(s.second.x, drift)]


class CuckerSmaleAlignment(PairInteraction):
    """Cucker-Smale alignment, `dv_i/dt = K / N sum_j psi(r_ij) (v_j - v_i)` with `psi(r) = (1 + r^2)^(-beta)`.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the group or particle of the first members.

        second:
            The path of the group or particle of the second members.

        cutoff:
            The distance beyond which agents do not align, large for the original all-to-all model.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

        K:
            The coupling strength.

        beta:
            The decay exponent of the communication weight.

        n_agents:
            The number of agents `N` that normalizes the coupling.

    Examples:
        >>> from skcomplex.models.physics.active_matter import CuckerSmaleAlignment
        >>> CuckerSmaleAlignment('align', first='birds', second='birds', beta=0.25).beta
        0.25
    """

    K: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    beta: float = parameter(default=0.25, domain=Domain.NONNEGATIVE)
    n_agents: int = 1

    def laws(self, s: Symbols) -> list[Law]:
        """Return the opposite velocity changes of the two members of a pair.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        separation = s.first.x - s.second.x
        weight = s.K / self.n_agents * (1 + separation.dot_product(separation)) ** (-s.beta)
        change = weight * (s.second.v - s.first.v)
        return [Rate(s.first.v, change), Rate(s.second.v, -change)]


class PolarFluid(Element):
    """Site of the Toner-Tu polar active fluid, with a density `rho` and a velocity `(vx, vy)`.

    The density is conserved, `drho/dt + div(rho v) = 0`, and the velocity follows
    `dv/dt + lam (v . grad) v = a (rho - rho_c) v - beta |v|^2 v - sigma grad(rho) + D laplacian(v)` with additive
    noise of intensity `noise`. Above the critical density the fluid moves with the speed
    `sqrt(a (rho - rho_c) / beta)`.

    Args:
        label:
            The label of the site or the field.

        rho:
            The initial density.

        vx:
            The initial velocity along the first axis.

        vy:
            The initial velocity along the second axis.

        a:
            The rate of the ordering instability per unit density.

        rho_c:
            The critical density of the ordering transition.

        beta:
            The saturation of the speed.

        lam:
            The strength of the self-advection.

        sigma:
            The pressure per unit density.

        D:
            The viscosity of the velocity.

        noise:
            The intensity of the velocity noise.

    Examples:
        >>> from skcomplex.models.physics.active_matter import PolarFluid
        >>> PolarFluid('site').rho_c
        0.5
    """

    rho: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    vx: float = state(default=0.0)
    vy: float = state(default=0.0)
    a: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    rho_c: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    beta: float = parameter(default=1.0, domain=Domain.POSITIVE)
    lam: float = parameter(default=1.0)
    sigma: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    D: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    noise: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the continuity equation and the Toner-Tu equation of each velocity component.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        velocity = [s.vx, s.vy]
        pressure = gradient(s.sigma * s.rho, 2)
        speed_squared = s.vx**2 + s.vy**2
        laws: list[Law] = [Rate(s.rho, -divergence([s.rho * s.vx, s.rho * s.vy]))]
        for axis, component in enumerate(velocity):
            transport = advection(component, [s.lam * item for item in velocity])
            growth = (s.a * (s.rho - s.rho_c) - s.beta * speed_squared) * component
            laws += [
                Rate(component, -transport + growth - pressure[axis] + s.D * laplacian(component)),
                Noise(component, sqrt(2 * s.noise)),
            ]
        return laws


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
    group = ActiveParticle.group('swimmers', n_particles, x=positions, theta=angles, speed=speed, D_t=0.0, D_r=D_r)
    return System(elements=[group], interactions=[PeriodicBox('box', target='swimmers', size=size)])


@register_model(
    field='physics/active_matter',
    tags=('active matter', 'sde', 'phase separation', 'pair interaction'),
    references=('Cates and Tailleur 2015', 'Bechinger et al. 2016', 'Fily and Marchetti 2012'),
    defaults={'until': 20.0, 'dt': 0.1, 'solver': EulerMaruyama(dt=0.005)},
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
    repulsion = SoftRepulsion('contact', first='swimmers', second='swimmers', cutoff=1.0, period=size, k=k)
    return System(elements=system.elements, interactions=[*system.interactions, repulsion], space=EuclideanSpace(2))


@register_model(
    field='physics/active_matter',
    tags=('active matter', 'sde', 'clusters', 'phoresis', 'pair interaction'),
    references=('Argun et al. 2021', 'Palacci et al. 2013'),
    defaults={'until': 20.0, 'dt': 0.1, 'solver': EulerMaruyama(dt=0.002)},
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
    repulsion = SoftRepulsion('contact', first='swimmers', second='swimmers', cutoff=2.0, period=size, k=k)
    attraction = PhoreticAttraction(
        'phoresis',
        first='swimmers',
        second='swimmers',
        cutoff=cutoff,
        period=size,
        v0=v0,
        R=1.0,
    )
    interactions = [*system.interactions, repulsion, attraction]
    return System(elements=system.elements, interactions=interactions, space=EuclideanSpace(2))


@register_model(
    field='physics/active_matter',
    tags=('flocking', 'ode', 'alignment', 'pair interaction'),
    references=('Cucker and Smale 2007', 'Vicsek and Zafeiris 2012', 'Bellomo et al. 2017'),
    defaults={'until': 20.0, 'dt': 0.5, 'solver': RK4(dt=0.05)},
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
            The model, with the positions `birds.x` and velocities `birds.v`.

    Examples:
        >>> from skcomplex.models.physics.active_matter import cucker_smale_flocking
        >>> cucker_smale_flocking(5).interactions[0].n_agents
        5
    """
    rng = np.random.default_rng(random_state)
    birds = Particle.group(
        'birds',
        n_agents,
        x=rng.uniform(0.0, size, (n_agents, 2)),
        v=rng.normal(0.0, 1.0, (n_agents, 2)),
        visual=VisualSpec('points', {'position': 'x', 'color_by': 'v', 'states': ()}),
    )
    alignment = CuckerSmaleAlignment(
        'align',
        first='birds',
        second='birds',
        cutoff=1e6,
        K=K,
        beta=beta,
        n_agents=n_agents,
    )
    return System(elements=[birds], interactions=[alignment], space=EuclideanSpace(2))


@register_model(
    field='physics/active_matter',
    tags=('field', 'flocking', 'hydrodynamics', 'pde'),
    references=('Toner and Tu 1995', 'Marchetti et al. 2013'),
    defaults={'until': 20.0, 'dt': 0.5, 'solver': EulerMaruyama(dt=0.02)},
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
        >>> toner_tu((8, 8)).elements[0].size
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
    return System(elements=[Field('flock', PolarFluid, space, values={**values, **parameters})])
