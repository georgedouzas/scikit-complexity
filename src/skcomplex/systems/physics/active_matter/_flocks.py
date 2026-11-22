"""Vicsek flocks in discrete time.

The module covers polar alignment, cohesive flocks, and polar particles with nematic alignment.

`MEMORY` holds the number of past headings a particle keeps, the largest delay of the alignment. `FIT_POINTS` holds the
fewest past headings a predicted alignment fits a line through.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, I, atan2, cos, exp, oo, sin, sqrt
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, Period, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.laws.utils import neighbors, uniform, where
from ....core.components.spaces import (
    BaseNeighborhood,
    BaseSpace,
    ContinuousSpace,
    NearestNeighborhood,
    RadiusNeighborhood,
)
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ParameterError
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ..common.particles import PropulsionStepInteraction

CLOSE = 0.2
REPULSION = 1e3
MEMORY = 25
FIT_POINTS = 2
HEADINGS = VisualSpec(
    'points',
    {'position': 'x', 'radius': 0.15, 'color_by': 'theta', 'color_range': (-math.pi, math.pi)},
)


class CohesiveParticleElement(Element):
    """Particles of a Grégoire-Chaté flock with the positions `x` and the headings `theta`.

    Args:
        label:
            The label of the particles.

        x:
            The initial positions, one row per particle or shared.

        theta:
            The initial headings, one per particle or shared.

        size:
            The number of particles, or `None` for one.

        space:
            The continuous space of the particles, or `None` for none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import CohesiveParticleElement
        >>> CohesiveParticleElement('bird', theta=1.0).theta
        1.0
    """

    visual = HEADINGS

    def __init__(
        self,
        label: str,
        *,
        x: object = (0.0, 0.0),
        theta: Annotated[object, Period(2 * math.pi)] = 0.0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.x = x
        self.theta = theta


class CohesiveAlignmentInteraction(Interaction):
    """Alignment with cohesion of a Grégoire-Chaté flock.

    The particles align, repel close neighbours, and attract distant ones.

    Every step the new heading is the direction of `alpha sum_j exp(i theta_j) + beta sum_j f(r_ij) e_ij + eta n_i xi`
    over the `n_i` particles within the unit distance, where `e_ij` points from `i` to `j`, `xi` is a random unit
    vector, and the force `f` is a strong repulsion below `r_c`, zero at `r_e`, and attractive up to `r_a`. It is a
    school of fish or a flock of birds that stays together.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particles, with the positions `x` and the headings `theta`.

        eta:
            The strength of the vectorial noise.

        alpha:
            The strength of the alignment.

        beta:
            The strength of the cohesion.

        r_c:
            The distance below which the neighbours repel strongly.

        r_e:
            The preferred distance.

        r_a:
            The distance of the strongest attraction.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import CohesiveAlignmentInteraction
        >>> CohesiveAlignmentInteraction('cohesion', target='flock').r_e
        0.5
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        eta: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        alpha: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        beta: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        r_c: Annotated[float, RealSet.open(0, oo)] = CLOSE,
        r_e: Annotated[float, RealSet.open(0, oo)] = 0.5,
        r_a: Annotated[float, RealSet.open(0, oo)] = 0.8,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.r_c = r_c
        self.r_e = r_e
        self.r_a = r_a

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the update of the heading of the target by alignment, cohesion, and vectorial noise.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        s, member = context, context.target
        near, others = RadiusNeighborhood(1.0, center=True), RadiusNeighborhood(1.0)
        dx, dy = member.other.x[0] - member.x[0], member.other.x[1] - member.x[1]
        distance = sqrt(dx**2 + dy**2)
        force = where(distance < s.r_c, -REPULSION, (distance - s.r_e) / (4 * (s.r_a - s.r_e)))
        count = neighbors(1, over=near)
        angle = uniform('noise', -math.pi, math.pi)
        along_x = (
            s.alpha * neighbors(cos(member.other.theta), over=near)
            + s.beta * neighbors(force * dx / distance, over=others)
            + s.eta * count * cos(angle)
        )
        along_y = (
            s.alpha * neighbors(sin(member.other.theta), over=near)
            + s.beta * neighbors(force * dy / distance, over=others)
            + s.eta * count * sin(angle)
        )
        self.laws_ = [UpdateLaw(member.theta, atan2(along_y, along_x))]


class NematicParticleElement(Element):
    """Polar particles with the positions `x`, the headings `theta`, and the noise amplitudes `eta`.

    `HeadingAlignmentInteraction` with `nematic=True` sets the new heading to the circular mean of the headings within
    the unit distance, itself included, each flipped to the side nearer the particle's own heading, plus a uniform
    angle in `[-eta pi, eta pi]`. A propulsion step moves the particle along its heading.

    Args:
        label:
            The label of the particles.

        x:
            The initial positions, one row per particle or shared.

        theta:
            The initial headings, one per particle or shared.

        eta:
            The amplitude of the noise, as a fraction of `pi`.

        size:
            The number of particles, or `None` for one.

        space:
            The continuous space of the particles, or `None` for none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import NematicParticleElement
        >>> NematicParticleElement('rod', eta=0.2).eta
        0.2
    """

    visual = HEADINGS

    def __init__(
        self,
        label: str,
        *,
        x: object = (0.0, 0.0),
        theta: Annotated[object, Period(2 * math.pi)] = 0.0,
        eta: object = 0.1,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.x = x
        self.theta = theta
        self.eta = eta


class VicsekParticleElement(Element):
    """Particles of a Vicsek flock with the positions `x`, the headings `theta`, and the noise amplitudes `eta`.

    `HeadingAlignmentInteraction` sets the new heading to the circular mean of the headings within the unit distance,
    itself included, plus a uniform angle in `[-eta pi, eta pi]`. A propulsion step moves the particle along its
    heading, which wraps around the circle. A delayed alignment reads past headings, which hold the initial heading
    before the start.

    Args:
        label:
            The label of the particles.

        x:
            The initial positions, one row per particle or shared.

        theta:
            The initial headings, one per particle or shared.

        eta:
            The amplitude of the noise, as a fraction of `pi`.

        size:
            The number of particles, or `None` for one.

        space:
            The continuous space of the particles, or `None` for none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import VicsekParticleElement
        >>> VicsekParticleElement('bird').eta
        0.1
    """

    visual = HEADINGS

    def __init__(
        self,
        label: str,
        *,
        x: object = (0.0, 0.0),
        theta: Annotated[object, Period(2 * math.pi)] = 0.0,
        eta: Annotated[object, RealSet.closed_open(0, oo)] = 0.1,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.x = x
        self.theta = theta
        self.eta = eta


class HeadingAlignmentInteraction(Interaction):
    """Vicsek alignment.

    Every particle turns to the circular mean heading of its neighbours plus noise.

    Every step the heading `theta` of every particle becomes the circular mean of the headings within the unit
    distance, itself included, plus a uniform angle in `[-eta pi, eta pi]` with the parameter `eta` of the particle.
    With `nematic`, every heading is flipped to the side nearer the particle's own heading first, as for particles
    with an axis rather than a direction. Another neighbourhood, such as the `k` nearest particles or a vision cone,
    replaces the unit disk.

    With a positive `latency` `h`, every particle reads the headings of its neighbours `h` steps ago. With a negative
    latency it reads their headings predicted `|h|` steps ahead, by a least-squares line through their last `memory`
    headings, unwrapped by taking every change between two steps as the smallest angle.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of particles.

        nematic:
            Whether every heading is flipped to the side nearer the particle's own heading.

        neighborhood:
            The neighbourhood of every particle, itself included, or `None` for the unit disk.

        latency:
            The number of steps by which the headings of the neighbours are read late, or predicted ahead when
            negative.

        memory:
            The number of past headings a prediction fits, at least 2.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import HeadingAlignmentInteraction
        >>> HeadingAlignmentInteraction('alignment', target='flock', nematic=True).nematic
        True
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        nematic: bool = False,
        neighborhood: BaseNeighborhood | None = None,
        latency: int = 0,
        memory: int = 5,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.nematic = nematic
        self.neighborhood = neighborhood
        self.latency = latency
        self.memory = memory

    def _get_heading(self, heading: Expression, t: Expression) -> Expression:
        """Return the heading of a neighbour, delayed or predicted."""
        if self.latency >= 0:
            return heading if self.latency == 0 else heading(t=t - self.latency)
        past = [heading if step == 0 else heading(t=t - step) for step in range(self.memory)]
        unwrapped = [past[0]]
        for step in range(1, self.memory):
            change = past[step - 1] - past[step]
            unwrapped.append(unwrapped[-1] - atan2(sin(change), cos(change)))
        times = [-step for step in range(self.memory)]
        mean_time = sum(times) / self.memory
        mean_heading = sum(unwrapped) / self.memory
        spread = sum((time - mean_time) ** 2 for time in times)
        slope = sum((time - mean_time) * (value - mean_heading) for time, value in zip(times, unwrapped, strict=True))
        return mean_heading + slope / spread * (-self.latency - mean_time)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the heading of every particle to the mean heading of its neighbours plus noise.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        member, other = context.target, context.target.other
        over = RadiusNeighborhood(1.0, center=True) if self.neighborhood is None else self.neighborhood
        side = where(cos(other.theta - member.theta) >= 0, 1, -1) if self.nematic else None
        heading = self._get_heading(other.theta, context.t)
        mean = neighbors(heading, over=over, reduce='circular-mean', weight=side)
        noise = uniform('noise', -member.eta * math.pi, member.eta * math.pi)
        self.laws_ = [UpdateLaw(member.theta, mean + noise)]


def _get_orders(group: str) -> list[Observable]:
    """Return the mean heading and axis of a group of flockers and their moduli.

    The moduli are the polar and nematic orders.
    """
    return [
        Observable('mean_heading', _get_member_heading, group=group),
        Observable('polarization', _get_polarization),
        Observable('mean_axis', _get_axis, group=group),
        Observable('nematic_order', _get_nematic_order),
    ]


def _get_member_heading(member: SimpleNamespace) -> Expression:
    """Return the unit vector `exp(i theta)` of a member's heading.

    The vector lies in the complex plane.
    """
    return exp(I * member.theta)


def _get_polarization(s: SimpleNamespace) -> Expression:
    """Return the modulus of the mean heading, the polar order parameter."""
    return VicsekSystem._get_modulus('mean_heading', s)


def _get_axis(member: SimpleNamespace) -> Expression:
    """Return `exp(2 i theta)`, the heading of a member as an axis without direction."""
    return exp(2 * I * member.theta)


def _get_nematic_order(s: SimpleNamespace) -> Expression:
    """Return the modulus of the mean axis, the nematic order parameter."""
    return VicsekSystem._get_modulus('mean_axis', s)


class VicsekSystem(BaseSystem):
    """The Vicsek system.

    Particles move at a constant speed along the mean heading of their neighbours.

    At every step particle `i` takes the direction of the sum of the headings of the particles within the unit
    distance, itself included, plus a uniform angle in `[-eta pi, eta pi]`, and moves `v0` along it in a periodic box.
    At low noise and high density the particles align into a flock, and the polar order parameter
    `|<exp(i theta)>|` grows from about `1 / sqrt(N)` to near 1. Raising the noise or lowering the density destroys
    the order.

    Two variants of Argun et al. (2021, Section 8.3) make the interaction non-metric or non-reciprocal. With
    `neighbors`, a particle aligns with its `neighbors` nearest particles and itself at any distance, a topological
    interaction that need not be reciprocal, and the clusters grow with `neighbors`. With `vision`, a particle sees only
    the particles within the unit distance whose direction from it makes an angle of at most `vision` with its heading,
    a vision cone that breaks reciprocity. The two variants do not combine, since the nearest neighbourhood of the core
    has no cone.

    With a positive `delay` `h`, the sensory delay of Section 8.2, a particle aligns with the headings its neighbours
    had `h` steps ago, `theta_j(n + 1) = <theta_k(n - h)> + noise`. Argun et al. report that a short delay stabilizes
    the clusters, and a long delay hinders them. A negative delay is a prediction: the headings of the neighbours `|h|`
    steps ahead are extrapolated by a least-squares line through their last `memory` headings, and it hinders
    clustering. The delay lies in `[-25, 25]`, and before the start the past headings equal the initial heading. The
    system has the positions `flock.x`, the headings `flock.theta`, and the observables `polarization` and
    `nematic_order`.

    Args:
        n_particles:
            The number of particles.

        size:
            The side of the periodic box, in units of the interaction radius.

        v0:
            The distance a particle moves in one step.

        eta:
            The amplitude of the noise, as a fraction of `pi`.

        seed:
            The seed of the uniform initial positions and headings.

        neighbors:
            The number of nearest particles a particle aligns with, or `None` for the particles within the unit
            distance.

        vision:
            The half-angle of the vision cone in `(0, pi]`, or `None` for the whole disk.

        delay:
            The delay of the alignment in steps, negative for a prediction.

        memory:
            The number of past headings a prediction fits, from 2 to 26.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ParameterError:
            Both `neighbors` and `vision` are given, or the delay or the memory exceeds the 25 kept headings.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import VicsekSystem
        >>> VicsekSystem(10).prepare().system_.elements[0].size
        10
        >>> type(VicsekSystem(10, neighbors=4).prepare().system_.interactions[1].neighborhood).__name__
        'NearestNeighborhood'
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'flocking', 'alignment', 'discrete time', 'neighborhood')
    references = ('Vicsek et al. 1995', 'Argun et al. 2021', 'Jensen 2023', 'Vicsek and Zafeiris 2012')
    until = 200.0
    dt = 1.0
    observables = tuple(_get_orders('flock'))

    @staticmethod
    def _get_modulus(name: str, s: SimpleNamespace) -> Expression:
        """Return the modulus of an observable of a system."""
        return abs(getattr(s, name))

    def __init__(
        self,
        n_particles: int = 400,
        *,
        size: float = 10.0,
        v0: float = 0.5,
        eta: float = 0.1,
        seed: int = 0,
        neighbors: int | None = None,
        vision: float | None = None,
        delay: int = 0,
        memory: int = 5,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.size = size
        self.v0 = v0
        self.eta = eta
        self.seed = seed
        self.neighbors = neighbors
        self.vision = vision
        self.delay = delay
        self.memory = memory

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of Vicsek particles in a periodic box.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.

        Raises:
            ParameterError:
                Both `neighbors` and `vision` are given, or the delay or the memory exceeds 25 past headings.
        """
        super()._prepare(context)
        if self.neighbors is not None and self.vision is not None:
            error_msg = 'A Vicsek flock aligns with its nearest neighbours or within a vision cone, not both.'
            raise ParameterError(error_msg)
        if abs(self.delay) > MEMORY or not FIT_POINTS <= self.memory <= MEMORY + 1:
            error_msg = (
                f'The delay {self.delay!r} and the memory {self.memory!r} need |delay| <= {MEMORY} and 2 <= memory <= '
                f'{MEMORY + 1}.'
            )
            raise ParameterError(error_msg)
        neighborhood: BaseNeighborhood | None = None
        if self.neighbors is not None:
            neighborhood = NearestNeighborhood(self.neighbors, center=True)
        if self.vision is not None:
            neighborhood = RadiusNeighborhood(1.0, angle=self.vision, direction='theta', center=True)
        rng = np.random.default_rng(self.seed)
        flock = VicsekParticleElement(
            'flock',
            x=rng.uniform(0.0, self.size, (self.n_particles, 2)),
            theta=rng.uniform(-math.pi, math.pi, self.n_particles),
            eta=self.eta,
            size=self.n_particles,
            space=ContinuousSpace(extent=(self.size, self.size), boundary='periodic'),
        )
        alignment = HeadingAlignmentInteraction(
            'alignment',
            target='flock',
            neighborhood=neighborhood,
            latency=self.delay,
            memory=self.memory,
        )
        self.system_ = System(
            self.label,
            elements=[flock],
            interactions=[PropulsionStepInteraction('propulsion', target='flock', speed=self.v0), alignment],
        )


class GregoireChateSystem(BaseSystem):
    """The Grégoire-Chaté system.

    Aligning particles also attract and repel, forming cohesive flocks.

    At every step particle `i` takes the direction of `alpha sum_j exp(i theta_j) + beta sum_j f(r_ij) e_ij + eta n_i
    xi`, over the `n_i` particles within the unit distance, where `e_ij` points from `i` to `j`, `xi` is a random unit
    vector, and the force `f` is a strong repulsion below `r_c = 0.2`, zero at the preferred distance `r_e = 0.5`, and
    attractive up to `r_a = 0.8`. The particle then moves `v0` along the new heading. At low noise and moderate cohesion
    the particles gather into cohesive, aligned flocks, and the polar order parameter grows towards 1. A strong cohesion
    locks them in place, and the order is lost. The system has the positions `flock.x`, the headings `flock.theta`, and
    the observables `polarization` and `nematic_order`.

    Args:
        n_particles:
            The number of particles.

        size:
            The side of the periodic box, in units of the interaction radius.

        v0:
            The distance a particle moves in one step.

        eta:
            The strength of the vectorial noise.

        alpha:
            The strength of the alignment.

        beta:
            The strength of the cohesion.

        seed:
            The seed of the uniform initial positions and headings.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import GregoireChateSystem
        >>> GregoireChateSystem(10, beta=2.0).prepare().system_.interactions[1].beta
        2.0
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'flocking', 'cohesion', 'discrete time', 'neighborhood')
    references = ('Grégoire and Chaté 2004', 'Vicsek and Zafeiris 2012')
    until = 300.0
    dt = 1.0
    observables = tuple(_get_orders('flock'))

    def __init__(
        self,
        n_particles: int = 200,
        *,
        size: float = 5.0,
        v0: float = 0.05,
        eta: float = 0.1,
        alpha: float = 1.0,
        beta: float = 0.5,
        seed: int = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.size = size
        self.v0 = v0
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.seed = seed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the Grégoire-Chaté system of aligning particles that also attract and repel.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        x, theta = GregoireChateSystem._get_flockers(self.n_particles, self.size, self.seed)
        flock = CohesiveParticleElement(
            'flock',
            x=x,
            theta=theta,
            size=self.n_particles,
            space=ContinuousSpace(extent=(self.size, self.size), boundary='periodic'),
        )
        cohesion = CohesiveAlignmentInteraction(
            'cohesion',
            target='flock',
            eta=self.eta,
            alpha=self.alpha,
            beta=self.beta,
        )
        self.system_ = System(
            self.label,
            elements=[flock],
            interactions=[PropulsionStepInteraction('propulsion', target='flock', speed=self.v0), cohesion],
        )

    @staticmethod
    def _get_flockers(n_particles: int, size: float, seed: int) -> tuple[Array, Array]:
        """Return uniform positions in a square and uniform headings."""
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, size, (n_particles, 2)), rng.uniform(-math.pi, math.pi, n_particles)


class GinelliNematicsSystem(BaseSystem):
    """Polar particles with nematic alignment.

    The particles order along a common axis but move both ways along it.

    At every step particle `i` takes the direction of `sum_j sign(cos(theta_j - theta_i)) exp(i theta_j)` over the
    particles within the unit distance, itself included, so it aligns with the axis of every neighbour on the side
    nearer its own heading, plus a uniform angle in `[-eta pi, eta pi]`, and moves `v0` along it. At low noise the
    particles share an axis, with as many moving one way as the other: the nematic order parameter `|<exp(2 i theta)>|`
    grows towards 1 while the polar order parameter stays small. The system has the positions `flock.x`, the headings
    `flock.theta`, and the observables `polarization` and `nematic_order`.

    Args:
        n_particles:
            The number of particles.

        size:
            The side of the periodic box, in units of the interaction radius.

        v0:
            The distance a particle moves in one step.

        eta:
            The amplitude of the noise, as a fraction of `pi`.

        seed:
            The seed of the uniform initial positions and headings.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import GinelliNematicsSystem
        >>> GinelliNematicsSystem(10).prepare().system_.elements[0].eta
        0.08
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'flocking', 'nematic', 'discrete time', 'neighborhood')
    references = ('Ginelli et al. 2010', 'Chaté 2020')
    until = 300.0
    dt = 1.0
    observables = tuple(_get_orders('flock'))

    def __init__(
        self,
        n_particles: int = 400,
        *,
        size: float = 10.0,
        v0: float = 0.3,
        eta: float = 0.08,
        seed: int = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.size = size
        self.v0 = v0
        self.eta = eta
        self.seed = seed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of polar particles with nematic alignment.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        x, theta = GregoireChateSystem._get_flockers(self.n_particles, self.size, self.seed)
        flock = NematicParticleElement(
            'flock',
            x=x,
            theta=theta,
            eta=self.eta,
            size=self.n_particles,
            space=ContinuousSpace(extent=(self.size, self.size), boundary='periodic'),
        )
        self.system_ = System(
            self.label,
            elements=[flock],
            interactions=[
                PropulsionStepInteraction('propulsion', target='flock', speed=self.v0),
                HeadingAlignmentInteraction('alignment', target='flock', nematic=True),
            ],
        )
