"""Collective active matter systems.

The module covers motility-induced phase separation, living crystals, flocking, and Toner-Tu.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement, GroupElement, PointElement
from ....core.components.interactions import PairInteraction
from ....core.components.laws import BaseLaw, NoiseLaw, RateLaw
from ....core.components.laws.utils import advection, divergence, gradient, laplacian
from ....core.components.spaces import ContinuousSpace, LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import EulerMaruyamaSolver, RK4Solver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..common.particles import ActiveParticleElement, SelfPropulsionInteraction


class PolarFluidElement(Element):
    """Site of the Toner-Tu polar active fluid.

    The site has a density `rho` and a velocity `(vx, vy)`.

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
        >>> from skcomplex.systems.physics.active_matter import PolarFluidElement
        >>> PolarFluidElement('site').rho_c
        0.5
    """

    law_classes = (
        NoiseLaw,
        RateLaw,
    )

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
        """Return the continuity equation and a Toner-Tu equation per velocity component.

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


class SoftSphereInteraction(PairInteraction):
    """Soft repulsion between overlapping spheres.

    The energy is `k * (cutoff - r) ** 2 / 2` closer than the cutoff.

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

        write:
            The name of the state whose rate the force changes, the position of order 2 by default.

        divisor:
            The name of the parameter that holds the masses of the members, or `None` for unit masses.

        period:
            The side of a periodic box, for minimum image distances, or `None` for the period of the space.

        k:
            The stiffness.

    Examples:
        >>> from skcomplex.systems.physics.active_matter._collective import SoftSphereInteraction
        >>> SoftSphereInteraction('contact', first='gas', second='gas', cutoff=0.1, k=50.0).k
        50.0
    """

    def __init__(
        self,
        label: str,
        first: str,
        second: str,
        cutoff: float = 1.0,
        position: str = 'x',
        *,
        write: str = 'x',
        divisor: str | None = None,
        period: float | None = None,
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
    ) -> None:
        super().__init__(
            label,
            first,
            second,
            cutoff=cutoff,
            position=position,
            write=write,
            potential=self._get_soft_sphere_energy,
            divisor=divisor,
            period=period,
        )
        self.k = k

    @staticmethod
    def _get_soft_sphere_energy(s: SimpleNamespace, r: Expression) -> Expression:
        """Return `k * (cutoff - r) ** 2 / 2`."""
        return s.k * (s.cutoff - r) ** 2 / 2


class CuckerSmaleAlignmentInteraction(PairInteraction):
    """Cucker-Smale alignment.

    The law is `x_i'' += K / N psi(r_ij) (x_j' - x_i')` with `psi(r) = (1 + r^2)^(-beta)`.

    Every pair of members within the cutoff pulls their velocities, the first derivatives of their positions of order
    2, towards each other with a weight that decays with their distance.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        first:
            The path of the group or particle of the first members.

        second:
            The path of the group or particle of the second members.

        cutoff:
            The distance beyond which pairs do not align, large for the all-to-all system.

        position:
            The name of the vector state of order 2 that holds the positions of the members.

        K:
            The strength of the alignment.

        beta:
            The decay exponent of the weight.

        period:
            The side of a periodic box, for minimum image distances, or `None` for the period of the space.

        n_agents:
            The number of agents `N` that normalizes the alignment.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import CuckerSmaleAlignmentInteraction
        >>> CuckerSmaleAlignmentInteraction('align', first='birds', second='birds', beta=0.25).beta
        0.25
    """

    def __init__(
        self,
        label: str,
        first: str,
        second: str,
        cutoff: float = 1.0,
        position: str = 'x',
        *,
        K: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.25,
        period: float | None = None,
        n_agents: int = 1,
    ) -> None:
        super().__init__(
            label,
            first,
            second,
            cutoff=cutoff,
            position=position,
            write=position,
            function=self._align,
            period=period,
        )
        self.K = K
        self.beta = beta
        self.n_agents = n_agents

    def _align(self, namespace: SimpleNamespace) -> Expression:
        """Return the pull of the velocity of the first member towards the second.

        The pull is `K / N psi(r) (x_j' - x_i')`.
        """
        first, second = namespace.first, namespace.second
        separation = getattr(first, self.position) - getattr(second, self.position)
        velocity = f'{self.position}_d1'
        weight = namespace.K / self.n_agents * (1 + separation.dot_product(separation)) ** (-namespace.beta)
        return weight * (getattr(second, velocity) - getattr(first, velocity))


class PhoreticAttractionInteraction(PairInteraction):
    """Phoretic attraction between active colloids.

    The attraction is the pair potential `-v0 R^2 / r` of an overdamped position.

    Every member of a pair drifts towards the other at the speed `v0 R^2 / r^2`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

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
            The side of a periodic box, for minimum image distances, or `None` for the period of the space.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import PhoreticAttractionInteraction
        >>> PhoreticAttractionInteraction('phoresis', first='colloids', second='colloids', v0=20.0).v0
        20.0
    """

    def __init__(
        self,
        label: str,
        first: str,
        second: str,
        cutoff: float = 1.0,
        position: str = 'x',
        *,
        v0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        R: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        period: float | None = None,
    ) -> None:
        super().__init__(
            label,
            first,
            second,
            cutoff=cutoff,
            position=position,
            write=position,
            potential=self._attraction,
            period=period,
        )
        self.v0 = v0
        self.R = R

    @staticmethod
    def _attraction(namespace: SimpleNamespace, r: Expression) -> Expression:
        """Return the potential `-v0 R^2 / r` of a pair at the distance `r`."""
        return -namespace.v0 * namespace.R**2 / r


def _disks(
    n_particles: int,
    size: float,
    random_state: int | np.random.Generator | None,
) -> tuple[Array, Array]:
    """Return positions on a jittered square grid over a box, and uniform orientations."""
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
) -> CompositeSystem:
    """Return active Brownian disks in a periodic box, without pair interactions."""
    positions, angles = _disks(n_particles, size, random_state)
    group = GroupElement(
        'swimmers',
        element=ActiveParticleElement,
        size=n_particles,
        values={'x': positions, 'theta': angles, 'D_t': 0.0, 'D_r': D_r},
    )
    propulsion = SelfPropulsionInteraction('propulsion', target='swimmers', speed=speed)
    return CompositeSystem(
        elements=[group],
        interactions=[propulsion],
        space=ContinuousSpace(extent=(size, size), boundary='periodic'),
    )


class MotilityInducedPhaseSeparationSystem(BaseSystem):
    """Repulsive active Brownian disks of unit diameter.

    The disks phase separate into clusters at high speed.

    Disks that collide head on block each other until their orientations diffuse away, so fast swimmers with slow
    rotational diffusion, a large Péclet number `v / D_r`, accumulate where they are slow and form dense clusters in a
    dilute gas. The system has the positions `swimmers.x` and orientations `swimmers.theta`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import MotilityInducedPhaseSeparationSystem
        >>> MotilityInducedPhaseSeparationSystem(16).prepare().system_.interactions[1].k
        50.0
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'sde', 'phase separation', 'pair interaction')
    references = ('Cates and Tailleur 2015', 'Bechinger et al. 2016', 'Fily and Marchetti 2012')
    until = 20.0
    dt = 0.1
    solver = EulerMaruyamaSolver(dt=0.005)

    def __init__(
        self,
        n_particles: int = 400,
        *,
        packing_fraction: float = 0.5,
        speed: float = 1.0,
        D_r: float = 0.01,
        k: float = 50.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_particles = n_particles
        self.packing_fraction = packing_fraction
        self.speed = speed
        self.D_r = D_r
        self.k = k
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return repulsive active Brownian disks of unit diameter.

        The disks phase separate into clusters at high speed.
        """
        size = math.sqrt(self.n_particles * math.pi / 4 / self.packing_fraction)
        system = _active_disks(self.n_particles, size, self.speed, self.D_r, self.random_state)
        repulsion = SoftSphereInteraction(
            'contact',
            first='swimmers',
            second='swimmers',
            cutoff=1.0,
            period=size,
            k=self.k,
        )
        return CompositeSystem(
            self.label,
            elements=system.elements,
            interactions=[*system.interactions, repulsion],
            space=system.space,
        )


class LivingCrystalsSystem(BaseSystem):
    """Active colloids of radius 1 with phoretic attraction.

    The colloids gather into clusters that slowly coarsen.

    The colloids swim as active Brownian particles, drift towards each other at the speed `v0 / r^2` within the cutoff,
    and a stiff soft repulsion stands in for the hard-sphere correction at contact. The system has the positions
    `swimmers.x` and orientations `swimmers.theta`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import LivingCrystalsSystem
        >>> LivingCrystalsSystem(16).prepare().system_.interactions[2].v0
        20.0
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'sde', 'clusters', 'phoresis', 'pair interaction')
    references = ('Argun et al. 2021', 'Palacci et al. 2013')
    until = 20.0
    dt = 0.1
    solver = EulerMaruyamaSolver(dt=0.002)

    def __init__(
        self,
        n_particles: int = 100,
        *,
        size: float = 50.0,
        speed: float = 3.0,
        D_r: float = 1.0,
        v0: float = 20.0,
        cutoff: float = 10.0,
        k: float = 200.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_particles = n_particles
        self.size = size
        self.speed = speed
        self.D_r = D_r
        self.v0 = v0
        self.cutoff = cutoff
        self.k = k
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        system = _active_disks(self.n_particles, self.size, self.speed, self.D_r, self.random_state)
        repulsion = SoftSphereInteraction(
            'contact',
            first='swimmers',
            second='swimmers',
            cutoff=2.0,
            period=self.size,
            k=self.k,
        )
        attraction = PhoreticAttractionInteraction(
            'phoresis',
            first='swimmers',
            second='swimmers',
            cutoff=self.cutoff,
            period=self.size,
            v0=self.v0,
            R=1.0,
        )
        interactions = [*system.interactions, repulsion, attraction]
        return CompositeSystem(self.label, elements=system.elements, interactions=interactions, space=system.space)


class CuckerSmaleFlockingSystem(BaseSystem):
    """Cucker-Smale flocking.

    Every agent aligns its velocity with all others, weighted by distance.

    The mean velocity is conserved, and for `beta <= 1/2` the velocities converge to it from any start. The system has
    the positions `birds.x` and velocities `birds.x_d1`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import CuckerSmaleFlockingSystem
        >>> CuckerSmaleFlockingSystem(5).prepare().system_.interactions[0].n_agents
        5
    """

    field = 'physics/active_matter'
    topics = ('flocking', 'ode', 'alignment', 'pair interaction')
    references = ('Cucker and Smale 2007', 'Vicsek and Zafeiris 2012', 'Bellomo et al. 2017')
    until = 20.0
    dt = 0.5
    solver = RK4Solver(dt=0.05)

    def __init__(
        self,
        n_agents: int = 50,
        *,
        K: float = 1.0,
        beta: float = 0.25,
        size: float = 10.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_agents = n_agents
        self.K = K
        self.beta = beta
        self.size = size
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return Cucker-Smale flocking.

        Every agent aligns its velocity with all others, weighted by distance.
        """
        rng = np.random.default_rng(self.random_state)
        birds = GroupElement(
            'birds',
            element=PointElement,
            size=self.n_agents,
            values={
                'x': rng.uniform(0.0, self.size, (self.n_agents, 2)),
                'x_d1': rng.normal(0.0, 1.0, (self.n_agents, 2)),
            },
            visual=VisualSpec('points', {'position': 'x', 'color_by': 'x_d1', 'states': ()}),
        )
        alignment = CuckerSmaleAlignmentInteraction(
            'align',
            first='birds',
            second='birds',
            cutoff=1e6,
            K=self.K,
            beta=self.beta,
            n_agents=self.n_agents,
        )
        return CompositeSystem(self.label, elements=[birds], interactions=[alignment], space=ContinuousSpace(n_dim=2))


class TonerTuSystem(BaseSystem):
    """The Toner-Tu hydrodynamics of a polar flock on a periodic lattice.

    The flock starts from a disordered velocity field.

    Above the critical density the velocity grows and aligns until the whole fluid moves together with the speed `sqrt(a
    (rho0 - rho_c) / beta)`. The system has the fields `flock.rho`, `flock.vx`, and `flock.vy`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import TonerTuSystem
        >>> TonerTuSystem((8, 8)).prepare().system_.elements[0].prepare().size_
        64
    """

    field = 'physics/active_matter'
    topics = ('field', 'flocking', 'hydrodynamics', 'pde')
    references = ('Toner and Tu 1995', 'Marchetti et al. 2013')
    until = 20.0
    dt = 0.5
    solver = EulerMaruyamaSolver(dt=0.02)

    def __init__(
        self,
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
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.rho0 = rho0
        self.rho_c = rho_c
        self.a = a
        self.beta = beta
        self.lam = lam
        self.sigma = sigma
        self.D = D
        self.spacing = spacing
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        shape = tuple(self.shape)
        rng = np.random.default_rng(self.random_state)
        values = {
            'rho': np.full(shape, self.rho0),
            'vx': rng.normal(0.0, 0.1, shape),
            'vy': rng.normal(0.0, 0.1, shape) + 0.05,
        }
        parameters = {
            'a': self.a,
            'rho_c': self.rho_c,
            'beta': self.beta,
            'lam': self.lam,
            'sigma': self.sigma,
            'D': self.D,
        }
        space = LatticeSpace(shape, spacing=self.spacing)
        return CompositeSystem(
            self.label,
            elements=[FieldElement('flock', element=PolarFluidElement, space=space, values={**values, **parameters})],
        )
