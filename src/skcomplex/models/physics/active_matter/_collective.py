"""Collective active matter models: motility-induced phase separation, living crystals, flocking, and Toner-Tu."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo, sqrt
from sage.sets.real_set import RealSet

from ....core.elements import Element, FieldElement, GroupElement, ParticleElement
from ....core.interactions import PairInteraction
from ....core.laws import BaseLaw, NoiseLaw, RateLaw
from ....core.laws.utils import advection, divergence, gradient, laplacian
from ....core.spaces import ContinuousSpace, LatticeSpace
from ....core.system import System
from ....engine.solvers import EulerMaruyamaSolver, RK4Solver
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model
from ._particles import ActiveParticleElement


class SoftRepulsionInteraction(PairInteraction):
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

        position:
            The name of the vector state that holds the positions of the members.

        k:
            The stiffness times the mobility.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

    Examples:
        >>> from skcomplex.models.physics.active_matter import SoftRepulsionInteraction
        >>> SoftRepulsionInteraction('contact', first='swimmers', second='swimmers', k=50.0).k
        50.0
    """

    def __init__(
        self,
        label: str,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        cutoff: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        position: str = 'x',
        *,
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
        period: float | None = None,
    ) -> None:
        super().__init__(label, first, second, cutoff=cutoff, position=position)
        self.k = k
        self.period = period

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
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
        return [RateLaw(s.first.x, push), RateLaw(s.second.x, -push)]


class PhoreticAttractionInteraction(PairInteraction):
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

        position:
            The name of the vector state that holds the positions of the members.

        v0:
            The drift speed at the distance `R`.

        R:
            The particle radius.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

    Examples:
        >>> from skcomplex.models.physics.active_matter import PhoreticAttractionInteraction
        >>> PhoreticAttractionInteraction('phoresis', first='colloids', second='colloids', v0=20.0).v0
        20.0
    """

    def __init__(
        self,
        label: str,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        cutoff: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        position: str = 'x',
        *,
        v0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        R: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        period: float | None = None,
    ) -> None:
        super().__init__(label, first, second, cutoff=cutoff, position=position)
        self.v0 = v0
        self.R = R
        self.period = period

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
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
        return [RateLaw(s.first.x, -drift), RateLaw(s.second.x, drift)]


class CuckerSmaleAlignmentInteraction(PairInteraction):
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

        position:
            The name of the vector state that holds the positions of the members.

        K:
            The coupling strength.

        beta:
            The decay exponent of the communication weight.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

        n_agents:
            The number of agents `N` that normalizes the coupling.

    Examples:
        >>> from skcomplex.models.physics.active_matter import CuckerSmaleAlignmentInteraction
        >>> CuckerSmaleAlignmentInteraction('align', first='birds', second='birds', beta=0.25).beta
        0.25
    """

    def __init__(
        self,
        label: str,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        cutoff: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        position: str = 'x',
        *,
        K: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.25,
        period: float | None = None,
        n_agents: int = 1,
    ) -> None:
        super().__init__(label, first, second, cutoff=cutoff, position=position)
        self.K = K
        self.beta = beta
        self.period = period
        self.n_agents = n_agents

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
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
        return [RateLaw(s.first.v, change), RateLaw(s.second.v, -change)]


class PolarFluidElement(Element):
    """Site of the Toner-Tu polar active fluid, with a density `rho` and a velocity `(vx, vy)`.

    The density is conserved, `drho/dt + div(rho v) = 0`, and the velocity follows
    `dv/dt + lam (v . grad) v = a (rho - rho_c) v - beta |v|^2 v - sigma grad(rho) + D laplacian(v)` with additive
    noise of intensity `noise`. Above the critical density the fluid moves with the speed
    `sqrt(a (rho - rho_c) / beta)`.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.models.physics.active_matter import PolarFluidElement
        >>> PolarFluidElement('site').rho_c
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        vx: Annotated[float, StateSpec()] = 0.0,
        vy: Annotated[float, StateSpec()] = 0.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        rho_c: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        lam: Annotated[float, ParameterSpec()] = 1.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        D: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        noise: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho
        self.vx = vx
        self.vy = vy
        self.a = a
        self.rho_c = rho_c
        self.beta = beta
        self.lam = lam
        self.sigma = sigma
        self.D = D
        self.noise = noise

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
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
        laws: list[BaseLaw] = [RateLaw(s.rho, -divergence([s.rho * s.vx, s.rho * s.vy]))]
        for axis, component in enumerate(velocity):
            transport = advection(component, [s.lam * item for item in velocity])
            growth = (s.a * (s.rho - s.rho_c) - s.beta * speed_squared) * component
            laws += [
                RateLaw(component, -transport + growth - pressure[axis] + s.D * laplacian(component)),
                NoiseLaw(component, sqrt(2 * s.noise)),
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
    group = GroupElement(
        'swimmers',
        element=ActiveParticleElement,
        size=n_particles,
        values={'x': positions, 'theta': angles, 'speed': speed, 'D_t': 0.0, 'D_r': D_r},
    )
    return System(elements=[group], space=ContinuousSpace(extent=(size, size), boundary='periodic'))


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
        >>> motility_induced_phase_separation(16).interactions[0].k
        50.0
    """
    size = math.sqrt(n_particles * math.pi / 4 / packing_fraction)
    system = _active_disks(n_particles, size, speed, D_r, random_state)
    repulsion = SoftRepulsionInteraction('contact', first='swimmers', second='swimmers', cutoff=1.0, period=size, k=k)
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
        >>> living_crystals(16).interactions[1].v0
        20.0
    """
    system = _active_disks(n_particles, size, speed, D_r, random_state)
    repulsion = SoftRepulsionInteraction('contact', first='swimmers', second='swimmers', cutoff=2.0, period=size, k=k)
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
            The model, with the positions `birds.x` and velocities `birds.v`.

    Examples:
        >>> from skcomplex.models.physics.active_matter import cucker_smale_flocking
        >>> cucker_smale_flocking(5).interactions[0].n_agents
        5
    """
    rng = np.random.default_rng(random_state)
    birds = GroupElement(
        'birds',
        element=ParticleElement,
        size=n_agents,
        values={'x': rng.uniform(0.0, size, (n_agents, 2)), 'v': rng.normal(0.0, 1.0, (n_agents, 2))},
        visual=VisualSpec('points', {'position': 'x', 'color_by': 'v', 'states': ()}),
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
        >>> toner_tu((8, 8)).elements[0].check().prepare().size_
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
