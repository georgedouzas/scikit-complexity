"""Vicsek flocks in discrete time: polar alignment, cohesive flocks, and polar particles with nematic alignment."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, I, atan2, cos, exp, oo, pi, sin, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import BaseInteraction, CouplingInteraction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import lag, neighbors, uniform, where
from ....core.components.spaces import (
    BaseNeighborhood,
    ConeNeighborhood,
    ContinuousSpace,
    DiscreteTime,
    NearestNeighborhood,
    RadiusNeighborhood,
)
from ....core.components.system import System
from ....core.components.types import Target
from ....core.errors import ParameterError
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system
from ._propulsion_step import PropulsionStepInteraction

CLOSE = 0.2


REPULSION = 1e3


class CohesiveParticleElement(Element):
    """Particle of a Grégoire-Chaté flock, which aligns, repels close neighbours, and attracts distant ones.

    The new heading is the direction of `alpha sum_j exp(i theta_j) + beta sum_j f(r_ij) e_ij + eta n_i xi` over the
    `n_i` particles within the unit distance, where `e_ij` points from `i` to `j`, `xi` is a random unit vector, and the
    force `f` is a strong repulsion below `r_c`, zero at `r_e`, and attractive up to `r_a`. A propulsion step moves the
    particle along its heading.

    Args:
        label:
            The label of the particle or the group.

        visual:
            How the particle is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        theta:
            The initial heading.

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

    Attributes:
        default_visual_:
            The visual of the particle when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import CohesiveParticleElement
        >>> CohesiveParticleElement('bird').r_e
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
        theta: Annotated[float, StateSpec(period=2 * pi, role='direction')] = 0.0,
        eta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        r_c: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = CLOSE,
        r_e: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        r_a: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.8,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.theta = theta
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.r_c = r_c
        self.r_e = r_e
        self.r_a = r_a

    def _attraction(self, distance: Expression, namespace: SimpleNamespace) -> Expression:
        """Return the cohesion force, repulsive below `r_c` and towards `r_e` beyond it."""
        s = namespace
        return where(distance < s.r_c, -REPULSION, (distance - s.r_e) / (4 * (s.r_a - s.r_e)))

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the alignment, the cohesion, and the vectorial noise of the heading.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with the neighbour `namespace.other`.

        Returns:
            laws:
                The update of the heading.
        """
        s = namespace
        near, others = RadiusNeighborhood(1.0, center=True), RadiusNeighborhood(1.0)
        dx, dy = s.other.x[0] - s.x[0], s.other.x[1] - s.x[1]
        distance = sqrt(dx**2 + dy**2)
        force = self._attraction(distance, s)
        count = neighbors(1, over=near)
        angle = uniform('noise', -math.pi, math.pi)
        along_x = (
            s.alpha * neighbors(cos(s.other.theta), over=near)
            + s.beta * neighbors(force * dx / distance, over=others)
            + s.eta * count * cos(angle)
        )
        along_y = (
            s.alpha * neighbors(sin(s.other.theta), over=near)
            + s.beta * neighbors(force * dy / distance, over=others)
            + s.eta * count * sin(angle)
        )
        return [UpdateLaw(s.theta, atan2(along_y, along_x))]


MEMORY = 25
"""The number of past headings a particle keeps, the largest delay of the alignment."""


class DelayedVicsekParticleElement(Element):
    """Particle of a delayed Vicsek flock, which keeps its past headings for the delayed or predicted alignment.

    The heading keeps its last 25 values, `theta_lag1` to `theta_lag25`, which a delayed alignment reads, and which
    hold the initial heading before the start. A propulsion step moves the particle along its current heading.

    Args:
        label:
            The label of the particle or the group.

        visual:
            How the particle is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        theta:
            The initial heading, also its value before the start.

        eta:
            The amplitude of the noise, as a fraction of `pi`.

    Attributes:
        default_visual_:
            The visual of the particle when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import DelayedVicsekParticleElement
        >>> DelayedVicsekParticleElement('bird').eta
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
        theta: Annotated[float, StateSpec(period=2 * pi, role='direction', lags=MEMORY)] = 0.0,
        eta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.theta = theta
        self.eta = eta


class NematicParticleElement(Element):
    """Polar particle with nematic alignment, which aligns with the axis of every neighbour on its nearer side.

    `HeadingAlignmentInteraction` with `nematic=True` sets the new heading to the circular mean of the headings within
    the unit distance, itself included, each flipped to the side nearer the particle's own heading, plus a uniform
    angle in `[-eta pi, eta pi]`. A propulsion step moves the particle along its heading.

    Args:
        label:
            The label of the particle or the group.

        visual:
            How the particle is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        theta:
            The initial heading.

        eta:
            The amplitude of the noise, as a fraction of `pi`.

    Attributes:
        default_visual_:
            The visual of the particle when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import NematicParticleElement
        >>> NematicParticleElement('rod', eta=0.2).eta
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
        theta: Annotated[float, StateSpec(period=2 * pi, role='direction')] = 0.0,
        eta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.theta = theta
        self.eta = eta


class VicsekParticleElement(Element):
    """Particle of a Vicsek flock, whose heading becomes the mean heading of its neighbours plus noise every step.

    `HeadingAlignmentInteraction` sets the new heading to the circular mean of the headings within the unit distance,
    itself included, plus a uniform angle in `[-eta pi, eta pi]`. A propulsion step moves the particle along its
    heading.

    Args:
        label:
            The label of the particle or the group.

        visual:
            How the particle is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        theta:
            The initial heading.

        eta:
            The amplitude of the noise, as a fraction of `pi`.

    Attributes:
        default_visual_:
            The visual of the particle when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import VicsekParticleElement
        >>> VicsekParticleElement('bird').eta
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
        theta: Annotated[float, StateSpec(period=2 * pi, role='direction')] = 0.0,
        eta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.theta = theta
        self.eta = eta


class HeadingAlignmentInteraction(CouplingInteraction):
    """Vicsek alignment, the turn of every particle to the circular mean heading of its neighbours plus noise.

    Every step the heading `theta` of every particle becomes the circular mean of the headings within the unit
    distance, itself included, plus a uniform angle in `[-eta pi, eta pi]` with the parameter `eta` of the particle.
    With `nematic`, every heading is flipped to the side nearer the particle's own heading first, as for particles
    with an axis rather than a direction. It is the map form of coupling with the circular mean over a radius. Another
    neighbourhood, such as the `k` nearest particles or a vision cone, replaces the unit disk.

    With a positive `latency` `h`, every particle reads the headings of its neighbours `h` steps ago. With a negative
    latency it reads their headings predicted `|h|` steps ahead, by a least-squares line through their last `memory`
    headings, unwrapped by taking every change between two steps as the smallest angle. A latent alignment needs
    particles that keep enough past headings, such as `DelayedVicsekParticleElement`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

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
        >>> HeadingAlignmentInteraction('alignment', target='flock', nematic=True).reduce
        'circular-mean'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        nematic: bool = False,
        neighborhood: BaseNeighborhood | None = None,
        latency: int = 0,
        memory: int = 5,
    ) -> None:
        super().__init__(
            label,
            target=target,
            read='theta',
            write='theta',
            function=self._get_heading,
            reduce='circular-mean',
            weights=self._get_side,
            form='map',
            combine=self._get_noisy,
            neighborhood=RadiusNeighborhood(1.0, center=True) if neighborhood is None else neighborhood,
        )
        self.nematic = nematic
        self.latency = latency
        self.memory = memory

    def _get_heading(self, namespace: SimpleNamespace) -> Expression:
        """Return the heading of the neighbour, delayed or predicted."""
        heading = namespace.other.theta
        if self.latency >= 0:
            return lag(heading, self.latency)
        unwrapped = [heading]
        for step in range(1, self.memory):
            change = lag(heading, step - 1) - lag(heading, step)
            unwrapped.append(unwrapped[-1] - atan2(sin(change), cos(change)))
        times = [-step for step in range(self.memory)]
        mean_time = sum(times) / self.memory
        mean_heading = sum(unwrapped) / self.memory
        spread = sum((time - mean_time) ** 2 for time in times)
        slope = sum((time - mean_time) * (value - mean_heading) for time, value in zip(times, unwrapped, strict=True))
        return mean_heading + slope / spread * (-self.latency - mean_time)

    def _get_side(self, namespace: SimpleNamespace) -> Expression:
        """Return -1 for a neighbour heading away from the particle when nematic, and 1 otherwise."""
        if not self.nematic:
            return SR(1)
        return where(cos(namespace.other.theta - namespace.member.theta) >= 0, 1, -1)

    @staticmethod
    def _get_noisy(namespace: SimpleNamespace, reduction: Target) -> Target:
        """Return the mean heading plus a uniform angle in `[-eta pi, eta pi]`."""
        eta = namespace.member.eta
        return reduction + uniform('noise', -eta * math.pi, eta * math.pi)


FIT_POINTS = 2
"""The fewest past headings a predicted alignment fits a line through."""

HEADINGS = VisualSpec(
    'points',
    {'position': 'x', 'radius': 0.15, 'color_by': 'theta', 'color_range': (-math.pi, math.pi)},
)


def _heading(member: SimpleNamespace) -> Expression:
    """Return the unit vector `exp(i theta)` of a member's heading in the complex plane."""
    return exp(I * member.theta)


def _axis(member: SimpleNamespace) -> Expression:
    """Return `exp(2 i theta)`, the heading of a member as an axis without direction."""
    return exp(2 * I * member.theta)


def _modulus(name: str, s: SimpleNamespace) -> Expression:
    """Return the modulus of an observable of a system."""
    return abs(getattr(s, name))


def _polarization(s: SimpleNamespace) -> Expression:
    """Return the modulus of the mean heading, the polar order parameter."""
    return _modulus('mean_heading', s)


def _nematic_order(s: SimpleNamespace) -> Expression:
    """Return the modulus of the mean axis, the nematic order parameter."""
    return _modulus('mean_axis', s)


def _orders(group: str) -> list[ObservableSpec]:
    """Return the mean heading and axis of a group of flockers and their moduli, the polar and nematic orders."""
    return [
        ObservableSpec('mean_heading', _heading, group=group),
        ObservableSpec('polarization', _polarization),
        ObservableSpec('mean_axis', _axis, group=group),
        ObservableSpec('nematic_order', _nematic_order),
    ]


def _flock(
    element: type[Element],
    n_particles: int,
    size: float,
    seed: int,
    *,
    v0: float,
    alignment: BaseInteraction | None = None,
    **values: float,
) -> System:
    """Return flockers of a type at uniform positions and headings in a periodic box that step `v0` every step."""
    rng = np.random.default_rng(seed)
    initial: dict[str, object] = {
        'x': rng.uniform(0.0, size, (n_particles, 2)),
        'theta': rng.uniform(-math.pi, math.pi, n_particles),
        **values,
    }
    flock = GroupElement('flock', element=element, size=n_particles, values=initial, visual=HEADINGS)
    return System(
        elements=[flock],
        interactions=[
            PropulsionStepInteraction('propulsion', target='flock', speed=v0),
            *([] if alignment is None else [alignment]),
        ],
        space=ContinuousSpace(extent=(size, size), boundary='periodic'),
        time=DiscreteTime(),
    )


@register_system(
    field='physics/active_matter',
    tags=('active matter', 'flocking', 'alignment', 'discrete time', 'neighborhood'),
    references=('Vicsek et al. 1995', 'Argun et al. 2021', 'Jensen 2023', 'Vicsek and Zafeiris 2012'),
    defaults={'until': 200.0, 'dt': 1.0},
    observables=lambda system: _orders('flock'),
)
def vicsek(
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
) -> System:
    """Return the Vicsek system: particles that move at a constant speed along the mean heading of their neighbours.

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
    clustering. Every particle then keeps its last 25 headings, so the delay lies in `[-25, 25]`, and before the start
    they equal the initial heading.

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

    Returns:
        system:
            The system, with the positions `flock.x`, the headings `flock.theta`, and the observables `polarization`
            and `nematic_order`.

    Raises:
        ParameterError:
            Both `neighbors` and `vision` are given, or the delay or the memory exceeds the 25 kept headings.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import vicsek
        >>> vicsek(10).elements[0].size
        10
        >>> type(vicsek(10, neighbors=4).interactions[1].neighborhood).__name__
        'NearestNeighborhood'
    """
    if neighbors is not None and vision is not None:
        error_msg = 'A Vicsek flock aligns with its nearest neighbours or within a vision cone, not both.'
        raise ParameterError(error_msg)
    if abs(delay) > MEMORY or not FIT_POINTS <= memory <= MEMORY + 1:
        error_msg = (
            f'The delay {delay!r} and the memory {memory!r} need |delay| <= {MEMORY} and 2 <= memory <= {MEMORY + 1}.'
        )
        raise ParameterError(error_msg)
    neighborhood: BaseNeighborhood | None = None
    if neighbors is not None:
        neighborhood = NearestNeighborhood(neighbors, center=True)
    if vision is not None:
        neighborhood = ConeNeighborhood(1.0, vision, direction='theta', center=True)
    alignment = HeadingAlignmentInteraction(
        'alignment',
        target='flock',
        neighborhood=neighborhood,
        latency=delay,
        memory=memory,
    )
    element = VicsekParticleElement if delay == 0 else DelayedVicsekParticleElement
    return _flock(element, n_particles, size, seed, v0=v0, alignment=alignment, eta=eta)


@register_system(
    field='physics/active_matter',
    tags=('active matter', 'flocking', 'cohesion', 'discrete time', 'neighborhood'),
    references=('Grégoire and Chaté 2004', 'Vicsek and Zafeiris 2012'),
    defaults={'until': 300.0, 'dt': 1.0},
    observables=lambda system: _orders('flock'),
)
def gregoire_chate(
    n_particles: int = 200,
    *,
    size: float = 5.0,
    v0: float = 0.05,
    eta: float = 0.1,
    alpha: float = 1.0,
    beta: float = 0.5,
    seed: int = 0,
) -> System:
    """Return the Grégoire-Chaté system: aligning particles that also attract and repel, forming cohesive flocks.

    At every step particle `i` takes the direction of `alpha sum_j exp(i theta_j) + beta sum_j f(r_ij) e_ij +
    eta n_i xi`, over the `n_i` particles within the unit distance, where `e_ij` points from `i` to `j`, `xi` is a
    random unit vector, and the force `f` is a strong repulsion below `r_c = 0.2`, zero at the preferred distance
    `r_e = 0.5`, and attractive up to `r_a = 0.8`. The particle then moves `v0` along the new heading. At low noise and
    moderate cohesion the particles gather into cohesive, aligned flocks, and the polar order parameter grows towards
    1. A strong cohesion locks them in place, and the order is lost.

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

    Returns:
        system:
            The system, with the positions `flock.x`, the headings `flock.theta`, and the observables `polarization`
            and `nematic_order`.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import gregoire_chate
        >>> gregoire_chate(10, beta=2.0).elements[0].values['beta']
        2.0
    """
    return _flock(CohesiveParticleElement, n_particles, size, seed, v0=v0, eta=eta, alpha=alpha, beta=beta)


@register_system(
    field='physics/active_matter',
    tags=('active matter', 'flocking', 'nematic', 'discrete time', 'neighborhood'),
    references=('Ginelli et al. 2010', 'Chaté 2020'),
    defaults={'until': 300.0, 'dt': 1.0},
    observables=lambda system: _orders('flock'),
)
def ginelli_nematics(
    n_particles: int = 400,
    *,
    size: float = 10.0,
    v0: float = 0.3,
    eta: float = 0.08,
    seed: int = 0,
) -> System:
    """Return polar particles with nematic alignment, which order along a common axis but move both ways along it.

    At every step particle `i` takes the direction of `sum_j sign(cos(theta_j - theta_i)) exp(i theta_j)` over the
    particles within the unit distance, itself included, so it aligns with the axis of every neighbour on the side
    nearer its own heading, plus a uniform angle in `[-eta pi, eta pi]`, and moves `v0` along it. At low noise the
    particles share an axis, with as many moving one way as the other: the nematic order parameter
    `|<exp(2 i theta)>|` grows towards 1 while the polar order parameter stays small.

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

    Returns:
        system:
            The system, with the positions `flock.x`, the headings `flock.theta`, and the observables `polarization`
            and `nematic_order`.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ginelli_nematics
        >>> ginelli_nematics(10).elements[0].values['eta']
        0.08
    """
    alignment = HeadingAlignmentInteraction('alignment', target='flock', nematic=True)
    return _flock(NematicParticleElement, n_particles, size, seed, v0=v0, alignment=alignment, eta=eta)
