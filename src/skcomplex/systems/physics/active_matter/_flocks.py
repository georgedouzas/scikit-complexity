"""Vicsek flocks in discrete time: polar alignment, cohesive flocks, and polar particles with nematic alignment."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression, I, exp

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import BaseInteraction
from ....core.components.spaces import (
    BaseNeighborhood,
    ConeNeighborhood,
    ContinuousSpace,
    DiscreteTime,
    NearestNeighborhood,
)
from ....core.components.system import System
from ....core.errors import ParameterError
from ....core.specs import ObservableSpec, VisualSpec
from ..._catalog import register_system
from ._cohesive_particle import CohesiveParticleElement
from ._delayed_vicsek_particle import MEMORY, DelayedVicsekParticleElement
from ._heading_alignment import HeadingAlignmentInteraction
from ._nematic_particle import NematicParticleElement
from ._propulsion_step import PropulsionStepInteraction
from ._vicsek_particle import VicsekParticleElement

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
