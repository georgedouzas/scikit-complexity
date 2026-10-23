"""Vicsek flocks in discrete time: polar alignment, cohesive flocks, and polar particles with nematic alignment."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np
from sage.all__sagemath_symbolics import I, atan2, cos, exp, sin, sqrt

from ....core.base import Domain, Expression, VisualSpec, parameter, state, vector
from ....core.components import Element
from ....core.laws import Law, Symbols, Update, neighbors, uniform, where
from ....core.runtime import register_model
from ....core.space import ContinuousSpace, DiscreteTime, Radius
from ....core.system import Observable, System

HEADINGS = VisualSpec(
    'points',
    {'position': 'x', 'radius': 0.15, 'color_by': 'theta', 'color_range': (-math.pi, math.pi)},
)
CLOSE = 0.2
REPULSION = 1e3


def _heading(member: Symbols) -> Expression:
    """Return the unit vector `exp(i theta)` of a member's heading in the complex plane."""
    return exp(I * member.theta)


def _axis(member: Symbols) -> Expression:
    """Return `exp(2 i theta)`, the heading of a member as an axis without direction."""
    return exp(2 * I * member.theta)


def _modulus(name: str, s: Symbols) -> Expression:
    """Return the modulus of an observable of a system."""
    return abs(getattr(s, name))


def _polarization(s: Symbols) -> Expression:
    """Return the modulus of the mean heading, the polar order parameter."""
    return _modulus('mean_heading', s)


def _nematic_order(s: Symbols) -> Expression:
    """Return the modulus of the mean axis, the nematic order parameter."""
    return _modulus('mean_axis', s)


def _orders(group: str) -> list[Observable]:
    """Return the mean heading and axis of a group of flockers and their moduli, the polar and nematic orders."""
    return [
        Observable('mean_heading', _heading, group=group),
        Observable('polarization', _polarization),
        Observable('mean_axis', _axis, group=group),
        Observable('nematic_order', _nematic_order),
    ]


def _step(s: Symbols) -> Law:
    """Return the move of a flocker by `v0` along its new heading."""
    return Update(s.x, s.x + s.v0 * vector([cos(s.theta), sin(s.theta)]))


def _vicsek_laws(s: Symbols) -> list[Law]:
    """Return the alignment with the mean heading within `r` plus angular noise, and the move."""
    within = Radius(1.0, center=True)
    turn = atan2(neighbors(sin(s.other.theta), over=within), neighbors(cos(s.other.theta), over=within))
    return [Update(s.theta, turn + uniform('noise', -s.eta * math.pi, s.eta * math.pi)), _step(s)]


def _attraction(distance: Expression, s: Symbols) -> Expression:
    """Return the cohesion force of the Grégoire-Chaté model, repulsive below `r_c` and towards `r_e` beyond it."""
    return where(distance < s.r_c, -REPULSION, (distance - s.r_e) / (4 * (s.r_a - s.r_e)))


def _cohesive_laws(s: Symbols) -> list[Law]:
    """Return the alignment, the cohesion, and the vectorial noise of the Grégoire-Chaté model, and the move."""
    near, others = Radius(1.0, center=True), Radius(1.0)
    dx, dy = s.other.x[0] - s.x[0], s.other.x[1] - s.x[1]
    distance = sqrt(dx**2 + dy**2)
    force = _attraction(distance, s)
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
    return [Update(s.theta, atan2(along_y, along_x)), _step(s)]


def _nematic_laws(s: Symbols) -> list[Law]:
    """Return the alignment with the axes of the neighbours, each flipped to the near side, plus noise, and the move."""
    within = Radius(1.0, center=True)
    side = where(cos(s.other.theta - s.theta) >= 0, 1, -1)
    turn = atan2(neighbors(side * sin(s.other.theta), over=within), neighbors(side * cos(s.other.theta), over=within))
    return [Update(s.theta, turn + uniform('noise', -s.eta * math.pi, s.eta * math.pi)), _step(s)]


FLOCKER_STATES = {
    'x': state(vector=True, position=True),
    'theta': state(default=0.0, domain=Domain.ANGLE),
}

VicsekParticle = Element.define(
    'VicsekParticle',
    states=FLOCKER_STATES,
    parameters={
        'v0': parameter(default=0.5, domain=Domain.NONNEGATIVE),
        'eta': parameter(default=0.1, domain=Domain.NONNEGATIVE),
    },
    laws=_vicsek_laws,
)

CohesiveParticle = Element.define(
    'CohesiveParticle',
    states=FLOCKER_STATES,
    parameters={
        'v0': parameter(default=0.05, domain=Domain.NONNEGATIVE),
        'eta': parameter(default=0.1, domain=Domain.NONNEGATIVE),
        'alpha': parameter(default=1.0, domain=Domain.NONNEGATIVE),
        'beta': parameter(default=1.0, domain=Domain.NONNEGATIVE),
        'r_c': parameter(default=CLOSE, domain=Domain.POSITIVE),
        'r_e': parameter(default=0.5, domain=Domain.POSITIVE),
        'r_a': parameter(default=0.8, domain=Domain.POSITIVE),
    },
    laws=_cohesive_laws,
)

NematicParticle = Element.define(
    'NematicParticle',
    states=FLOCKER_STATES,
    parameters={
        'v0': parameter(default=0.5, domain=Domain.NONNEGATIVE),
        'eta': parameter(default=0.1, domain=Domain.NONNEGATIVE),
    },
    laws=_nematic_laws,
)


def _flock(element: type[Element], n_particles: int, size: float, seed: int, **values: float) -> System:
    """Return flockers of a type at uniform positions and headings in a periodic box, under discrete time."""
    rng = np.random.default_rng(seed)
    flock = element.group(
        'flock',
        n_particles,
        x=rng.uniform(0.0, size, (n_particles, 2)),
        theta=rng.uniform(-math.pi, math.pi, n_particles),
        visual=HEADINGS,
        **values,
    )
    return System(components=[flock], space=ContinuousSpace(size, size, boundary='periodic'), time=DiscreteTime())


@register_model(
    field='physics/active_matter',
    tags=('active matter', 'flocking', 'alignment', 'discrete time', 'neighbourhood'),
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
) -> System:
    """Return the Vicsek model: particles that move at a constant speed along the mean heading of their neighbours.

    At every step particle `i` takes the direction of the sum of the headings of the particles within the unit
    distance, itself included, plus a uniform angle in `[-eta pi, eta pi]`, and moves `v0` along it in a periodic box.
    At low noise and high density the particles align into a flock, and the polar order parameter
    `|<exp(i theta)>|` grows from about `1 / sqrt(N)` to near 1. Raising the noise or lowering the density destroys
    the order.

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
            The model, with the positions `flock.x`, the headings `flock.theta`, and the observables `polarization`
            and `nematic_order`.

    Examples:
        >>> from skcomplex.models.physics.active_matter import vicsek
        >>> vicsek(10).components[0].size
        10
    """
    return _flock(VicsekParticle, n_particles, size, seed, v0=v0, eta=eta)


@register_model(
    field='physics/active_matter',
    tags=('active matter', 'flocking', 'cohesion', 'discrete time', 'neighbourhood'),
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
    """Return the Grégoire-Chaté model: aligning particles that also attract and repel, forming cohesive flocks.

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
            The model, with the positions `flock.x`, the headings `flock.theta`, and the observables `polarization`
            and `nematic_order`.

    Examples:
        >>> from skcomplex.models.physics.active_matter import gregoire_chate
        >>> gregoire_chate(10, beta=2.0).components[0].values['beta']
        2.0
    """
    return _flock(CohesiveParticle, n_particles, size, seed, v0=v0, eta=eta, alpha=alpha, beta=beta)


@register_model(
    field='physics/active_matter',
    tags=('active matter', 'flocking', 'nematic', 'discrete time', 'neighbourhood'),
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
            The model, with the positions `flock.x`, the headings `flock.theta`, and the observables `polarization`
            and `nematic_order`.

    Examples:
        >>> from skcomplex.models.physics.active_matter import ginelli_nematics
        >>> ginelli_nematics(10).components[0].values['eta']
        0.08
    """
    return _flock(NematicParticle, n_particles, size, seed, v0=v0, eta=eta)
