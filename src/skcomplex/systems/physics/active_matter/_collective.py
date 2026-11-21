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
from sage.all__sagemath_symbolics import SR, Expression, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import BaseInteraction, Interaction
from ....core.components.laws import BaseLaw, NoiseLaw, RateLaw
from ....core.components.laws.utils import advection, divergence, gradient, laplacian, neighbors
from ....core.components.spaces import BaseSpace, ContinuousSpace, LatticeSpace, RadiusNeighborhood
from ....core.components.types import Target
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, EulerMaruyamaSolver, RK4Solver
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ...general.elements import PointElement
from ..common.particles import ActiveDiffusionInteraction, ActiveParticleElement, SelfPropulsionInteraction


class PolarFluidElement(Element):
    """Sites of a polar active fluid with the density `rho` and the velocity `(vx, vy)`.

    Args:
        label:
            The label of the fluid.

        rho:
            The initial density, one per site or shared.

        vx:
            The initial velocity along the first axis, one per site or shared.

        vy:
            The initial velocity along the second axis, one per site or shared.

        space:
            The lattice of the sites, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import PolarFluidElement
        >>> PolarFluidElement('site', rho=2.0).rho
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        rho: object = 1.0,
        vx: object = 0.0,
        vy: object = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.rho = rho
        self.vx = vx
        self.vy = vy


class TonerTuInteraction(Interaction):
    """Toner-Tu hydrodynamics of a polar active fluid.

    The density is conserved, `drho/dt + div(rho v) = 0`, and the velocity follows
    `dv/dt + lam (v . grad) v = a (rho - rho_c) v - beta |v|^2 v - sigma grad(rho) + D laplacian(v)` with additive
    noise of intensity `noise`. Above the critical density the fluid moves with the speed
    `sqrt(a (rho - rho_c) / beta)`. It is a flock of birds, a school of fish, or a swarm of bacteria seen from afar.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the fluid, with the density `rho` and the velocity `(vx, vy)`.

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
        >>> from skcomplex.systems.physics.active_matter import TonerTuInteraction
        >>> TonerTuInteraction('flow', target='flock').rho_c
        0.5
    """

    law_classes = (NoiseLaw, RateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        a: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        rho_c: Annotated[float, RealSet.closed_open(0, oo)] = 0.5,
        beta: Annotated[float, RealSet.open(0, oo)] = 1.0,
        lam: float = 1.0,
        sigma: Annotated[float, RealSet.closed_open(0, oo)] = 0.5,
        D: Annotated[float, RealSet.closed_open(0, oo)] = 0.5,
        noise: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.a = a
        self.rho_c = rho_c
        self.beta = beta
        self.lam = lam
        self.sigma = sigma
        self.D = D
        self.noise = noise

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the continuity equation and a Toner-Tu equation per velocity component of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            rho, vx, vy = s.target.rho, s.target.vx, s.target.vy
            velocity = [vx, vy]
            pressure = gradient(s.sigma * rho, 2)
            speed_squared = vx**2 + vy**2
            laws: list[BaseLaw] = [RateLaw(rho.diff(s.t), -divergence([rho * vx, rho * vy]))]
            for axis, component in enumerate(velocity):
                transport = advection(component, [s.lam * item for item in velocity])
                growth = (s.a * (rho - s.rho_c) - s.beta * speed_squared) * component
                laws += [
                    RateLaw(component.diff(s.t), -transport + growth - pressure[axis] + s.D * laplacian(component)),
                    NoiseLaw(component, sqrt(2 * s.noise)),
                ]
            self.laws_ = laws


class SoftSphereInteraction(Interaction):
    """Soft repulsion between overlapping spheres.

    The energy is `k (cutoff - r)^2 / 2` closer than the cutoff, so every member is pushed by `k (cutoff - r)` along the
    line of centres from each member that overlaps it, `x_i' += sum_j k (cutoff - r_ij) (x_i - x_j) / r_ij`. In a
    periodic box the separations are minimum images. The push moves an overdamped sphere, and a subclass for spheres
    with inertia overrides the hook `_get_rate_law` to put it on the acceleration, `RateLaw(x.diff(t, 2), push)`. It is
    the contact of disks in a granular layer, of cells in a tissue, or of swimmers that bump into each other.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of spheres.

        cutoff:
            The contact distance, the sum of the two radii.

        k:
            The stiffness.

    Examples:
        >>> from skcomplex.systems.physics.active_matter._collective import SoftSphereInteraction
        >>> SoftSphereInteraction('contact', target='gas', cutoff=0.1, k=50.0).k
        50.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        cutoff: Annotated[float, RealSet.open(0, oo)] = 1.0,
        k: Annotated[float, RealSet.open(0, oo)] = 100.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.cutoff = cutoff
        self.k = k

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the push of the overlapping neighbours on every sphere.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, x = context, context.target.x
            separation = x - s.target.other.x
            r = sqrt(separation.dot_product(separation))
            distance = SR.var('r', domain='positive')
            slope = (s.k * (s.cutoff - distance) ** 2 / 2).diff(distance).subs({distance: r})
            push = neighbors(-slope * separation / r, RadiusNeighborhood(self.cutoff))
            self.laws_ = [self._get_rate_law(x, push, s.t)]

    def _get_rate_law(self, x: Target, push: object, t: Expression) -> RateLaw:
        """Return the law of the push on the velocity of a sphere, `x' = push`.

        This is the hook a subclass overrides for spheres with inertia, `RateLaw(x.diff(t, 2), push)`.

        Args:
            x:
                The position of the sphere as a function of the time.

            push:
                The sum of the pushes of the overlapping neighbours.

            t:
                The time.

        Returns:
            law:
                The rate law of the position.
        """
        return RateLaw(x.diff(t), push)


class CuckerSmaleAlignmentInteraction(Interaction):
    """Cucker-Smale alignment, `x_i'' += K / N sum_j psi(r_ij) (x_j' - x_i')` with `psi(r) = (1 + r^2)^(-beta)`.

    Every member pulls its velocity `x'` towards those of the members within the cutoff, with a weight that decays with
    their distance. It is a flock of birds, a school of fish, or a fleet of drones that agree on a heading.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of agents.

        cutoff:
            The distance beyond which pairs do not align, large for the all-to-all system.

        K:
            The strength of the alignment.

        beta:
            The decay exponent of the weight.

        n_agents:
            The number of agents `N` that normalizes the alignment.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import CuckerSmaleAlignmentInteraction
        >>> CuckerSmaleAlignmentInteraction('align', target='birds', beta=0.25).beta
        0.25
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        cutoff: float = 1.0,
        K: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        beta: Annotated[float, RealSet.closed_open(0, oo)] = 0.25,
        n_agents: int = 1,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.cutoff = cutoff
        self.K = K
        self.beta = beta
        self.n_agents = n_agents

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the pull of the velocity of every agent towards those of the others.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context.target, context.t
            separation = s.x - s.other.x
            weight = (1 + separation.dot_product(separation)) ** (-context.beta)
            pull = neighbors(weight * (s.other.x.diff(t) - s.x.diff(t)), over=RadiusNeighborhood(self.cutoff))
            self.laws_ = [RateLaw(s.x.diff(t, 2), context.K / self.n_agents * pull)]


class PhoreticAttractionInteraction(Interaction):
    """Phoretic attraction between active colloids, the pair potential `-v0 R^2 / r`.

    Every member of a pair closer than the cutoff drifts towards the other at the speed `v0 R^2 / r^2`. It is a
    light-activated colloid that gathers others by the chemicals it releases.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the colloids, with the positions `x`.

        v0:
            The drift speed at the distance `R`.

        R:
            The particle radius.

        cutoff:
            The distance beyond which the attraction is neglected.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import PhoreticAttractionInteraction
        >>> PhoreticAttractionInteraction('phoresis', target='colloids', v0=20.0).v0
        20.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        v0: Annotated[float, RealSet.closed_open(0, oo)] = 20.0,
        R: Annotated[float, RealSet.open(0, oo)] = 1.0,
        cutoff: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.v0 = v0
        self.R = R
        self.cutoff = cutoff

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the drift of every colloid of the target towards its neighbours within the cutoff.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            x = s.target.x
            separation = x - s.target.other.x
            r = sqrt(separation.dot_product(separation))
            drift = -s.v0 * s.R**2 / r**2 * separation / r
            self.laws_ = [RateLaw(x, neighbors(drift, RadiusNeighborhood(self.cutoff)))]


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import MotilityInducedPhaseSeparationSystem
        >>> MotilityInducedPhaseSeparationSystem(16).prepare().system_.interactions[2].k
        50.0
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'sde', 'phase separation', 'pair interaction')
    references = ('Cates and Tailleur 2015', 'Bechinger et al. 2016', 'Fily and Marchetti 2012')
    until = 20.0
    dt = 0.1

    @staticmethod
    def _get_disks(
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

    @staticmethod
    def _get_active_disks(
        n_particles: int,
        size: float,
        speed: float,
        D_r: float,
        random_state: int | np.random.Generator | None,
    ) -> tuple[ActiveParticleElement, list[BaseInteraction]]:
        """Return active Brownian disks in a periodic box, and their diffusion and propulsion."""
        positions, angles = MotilityInducedPhaseSeparationSystem._get_disks(n_particles, size, random_state)
        swimmers = ActiveParticleElement(
            'swimmers',
            x=positions,
            theta=angles,
            size=n_particles,
            space=ContinuousSpace(extent=(size, size), boundary='periodic'),
        )
        interactions: list[BaseInteraction] = [
            ActiveDiffusionInteraction('diffusion', target='swimmers', D_t=0.0, D_r=D_r),
            SelfPropulsionInteraction('propulsion', target='swimmers', speed=speed),
        ]
        return swimmers, interactions

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.packing_fraction = packing_fraction
        self.speed = speed
        self.D_r = D_r
        self.k = k
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of repulsive active Brownian disks of unit diameter.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        size = math.sqrt(self.n_particles * math.pi / 4 / self.packing_fraction)
        swimmers, interactions = MotilityInducedPhaseSeparationSystem._get_active_disks(
            self.n_particles,
            size,
            self.speed,
            self.D_r,
            self.random_state,
        )
        repulsion = SoftSphereInteraction('contact', target='swimmers', cutoff=1.0, k=self.k)
        self.system_ = System(
            self.label,
            elements=[swimmers],
            interactions=[*interactions, repulsion],
            solver=EulerMaruyamaSolver(dt=0.005),
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import LivingCrystalsSystem
        >>> LivingCrystalsSystem(16).prepare().system_.interactions[3].v0
        20.0
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'sde', 'clusters', 'phoresis', 'pair interaction')
    references = ('Argun et al. 2021', 'Palacci et al. 2013')
    until = 20.0
    dt = 0.1

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.size = size
        self.speed = speed
        self.D_r = D_r
        self.v0 = v0
        self.cutoff = cutoff
        self.k = k
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of active colloids with phoretic attraction.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        swimmers, interactions = MotilityInducedPhaseSeparationSystem._get_active_disks(
            self.n_particles,
            self.size,
            self.speed,
            self.D_r,
            self.random_state,
        )
        repulsion = SoftSphereInteraction('contact', target='swimmers', cutoff=2.0, k=self.k)
        attraction = PhoreticAttractionInteraction(
            'phoresis',
            target='swimmers',
            v0=self.v0,
            R=1.0,
            cutoff=self.cutoff,
        )
        self.system_ = System(
            self.label,
            elements=[swimmers],
            interactions=[*interactions, repulsion, attraction],
            solver=EulerMaruyamaSolver(dt=0.002),
        )


class BirdElement(PointElement):
    """Point drawn at its position and coloured by its velocity.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import BirdElement
        >>> BirdElement('bird', x=[0.0, 1.0]).x
        [0.0, 1.0]
    """

    visual = VisualSpec('points', {'position': 'x', 'color_by': "x'", 'states': ()})


class CuckerSmaleFlockingSystem(BaseSystem):
    """Cucker-Smale flocking.

    Every agent aligns its velocity with all others, weighted by distance.

    The mean velocity is conserved, and for `beta <= 1/2` the velocities converge to it from any start. The system has
    the positions `birds.x` and velocities `birds.x'`.

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

        solver:
            The solver of the problem, or `'auto'` for Runge-Kutta 4 with the step 0.05.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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

    def __init__(
        self,
        n_agents: int = 50,
        *,
        K: float = 1.0,
        beta: float = 0.25,
        size: float = 10.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_agents = n_agents
        self.K = K
        self.beta = beta
        self.size = size
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to agents at random places and velocities that align with all others.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        plane = ContinuousSpace(n_dim=2)
        birds = BirdElement(
            'birds',
            x=rng.uniform(0.0, self.size, (self.n_agents, 2)),
            values={"x'": rng.normal(0.0, 1.0, (self.n_agents, 2))},
            size=self.n_agents,
            space=plane,
        )
        alignment = CuckerSmaleAlignmentInteraction(
            'align',
            target='birds',
            cutoff=1e6,
            K=self.K,
            beta=self.beta,
            n_agents=self.n_agents,
        )
        self.system_ = System(
            self.label,
            elements=[birds],
            interactions=[alignment],
            space=plane,
            solver=RK4Solver(dt=0.05),
        )


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import TonerTuSystem
        >>> TonerTuSystem((8, 8)).prepare().simulate(until=1.0).get_values('flock.rho').shape
        (3, 64)
    """

    field = 'physics/active_matter'
    topics = ('field', 'flocking', 'hydrodynamics', 'pde')
    references = ('Toner and Tu 1995', 'Marchetti et al. 2013')
    until = 20.0
    dt = 0.5

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
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

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the Toner-Tu hydrodynamics of a polar flock on a periodic lattice.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        shape = tuple(self.shape)
        rng = np.random.default_rng(self.random_state)
        flock = PolarFluidElement(
            'flock',
            rho=np.full(shape, self.rho0).ravel(),
            vx=rng.normal(0.0, 0.1, shape).ravel(),
            vy=(rng.normal(0.0, 0.1, shape) + 0.05).ravel(),
            space=LatticeSpace(shape, spacing=self.spacing),
        )
        flow = TonerTuInteraction(
            'flow',
            target='flock',
            a=self.a,
            rho_c=self.rho_c,
            beta=self.beta,
            lam=self.lam,
            sigma=self.sigma,
            D=self.D,
        )
        self.system_ = System(self.label, elements=[flock], interactions=[flow], solver=EulerMaruyamaSolver(dt=0.02))
