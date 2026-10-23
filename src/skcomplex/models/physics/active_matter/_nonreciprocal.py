"""Nonreciprocal flocking: two species of Vicsek movers whose alignment with each other differs between the two."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import I, atan2, cos, exp, sin

from ....core.base import Domain, Expression, VisualSpec, parameter, state, vector
from ....core.components import Element
from ....core.laws import Law, Symbols, Update, neighbors, uniform
from ....core.runtime import register_model
from ....core.space import ContinuousSpace, DiscreteTime, Radius
from ....core.system import Observable, System

SPECIES = VisualSpec('points', {'position': 'x', 'radius': 0.04, 'color_by': 'species', 'color_range': (0.0, 1.0)})


def _coupling(s: Symbols) -> Expression:
    """Return the alignment of a member with a neighbour, by the species of both."""
    first, second = s.species, s.other.species
    return (
        s.J_aa * (1 - first) * (1 - second)
        + s.J_ab * (1 - first) * second
        + s.J_ba * first * (1 - second)
        + s.J_bb * first * second
    )


def _heading_a(member: Symbols) -> Expression:
    """Return the heading of a member of species A, doubled so the mean over both species is the mean over A."""
    return 2 * (1 - member.species) * exp(I * member.theta)


def _heading_b(member: Symbols) -> Expression:
    """Return the heading of a member of species B, doubled so the mean over both species is the mean over B."""
    return 2 * member.species * exp(I * member.theta)


class NonreciprocalMover(Element):
    """Vicsek mover of species A (0) or B (1), which aligns with the neighbours within `radius` by species couplings.

    Every step the heading becomes the direction of `sum_j J(species_i, species_j) exp(i theta_j)` over the movers
    within the radius, itself included, plus uniform noise of width `eta`, and the mover advances by `v0` along it.
    `J_aa` and `J_bb` align a species with itself, and `J_ab` and `J_ba` align A with B and B with A, so
    `J_ab != J_ba` makes the alignment nonreciprocal.

    Args:
        label:
            The label of the mover.

        x:
            The position.

        theta:
            The heading.

        species:
            The species, 0 for A or 1 for B.

        v0:
            The speed.

        eta:
            The width of the uniform noise of the heading.

        J_aa:
            The alignment of A with A.

        J_ab:
            The alignment of A with B.

        J_ba:
            The alignment of B with A.

        J_bb:
            The alignment of B with B.

        radius:
            The radius of the neighbourhood.

    Examples:
        >>> from skcomplex.models.physics.active_matter import NonreciprocalMover
        >>> NonreciprocalMover('mover', J_ba=-1.0).J_ba
        -1.0
    """

    visual: ClassVar[VisualSpec] = SPECIES
    x: npt.ArrayLike = state(default=0.0, vector=True, position=True)
    theta: float = state(default=0.0, domain=Domain.ANGLE)
    species: float = state(default=0.0, domain=Domain.BOOLEAN)
    v0: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)
    eta: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)
    J_aa: float = parameter(default=1.0)
    J_ab: float = parameter(default=1.0)
    J_ba: float = parameter(default=1.0)
    J_bb: float = parameter(default=1.0)
    radius: float = 1.0

    def laws(self, s: Symbols) -> list[Law]:
        """Return the alignment of the heading with the species-weighted neighbours and the step along it.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        over = Radius(self.radius, center=True)
        weight = _coupling(s)
        direction = atan2(
            neighbors(weight * sin(s.other.theta), over=over),
            neighbors(weight * cos(s.other.theta), over=over),
        )
        return [
            Update(s.theta, direction + uniform('noise', -s.eta / 2, s.eta / 2)),
            Update(s.x, s.x + s.v0 * vector([cos(s.theta), sin(s.theta)])),
        ]


@register_model(
    field='physics/active_matter',
    tags=('active matter', 'flocking', 'nonreciprocal', 'vicsek', 'chiral phase'),
    references=('Fruchart, Hanai, Littlewood, and Vitelli 2021', 'te Vrugt and Wittkowski 2024'),
    defaults={'until': 300.0, 'dt': 1.0},
    observables=lambda system: [
        Observable('heading_a', _heading_a, group='movers'),
        Observable('heading_b', _heading_b, group='movers'),
    ],
)
def nonreciprocal_vicsek(
    n_movers: int = 400,
    *,
    size: float = 5.0,
    J_ab: float = 1.0,
    J_ba: float = -1.0,
    eta: float = 0.3,
    random_state: int | np.random.Generator | None = 1,
) -> System:
    """Return two species of Vicsek movers in a periodic box whose alignment with each other is nonreciprocal.

    Half the movers are species A and half species B, each aligning with its own species. With `J_ab = J_ba` the two
    species flock together. With `J_ab = -J_ba`, A aligns with B while B antialigns with A, so neither settles: the
    mean headings of the species lock at a quarter turn from each other and rotate together, the chiral phase of
    Fruchart et al. (2021), which breaks the time-reversal symmetry that reciprocal alignment keeps.

    Args:
        n_movers:
            The number of movers, an even number.

        size:
            The side of the periodic box.

        J_ab:
            The alignment of A with B.

        J_ba:
            The alignment of B with A.

        eta:
            The width of the uniform noise of the headings.

        random_state:
            The seed or generator of the initial positions and headings.

    Returns:
        system:
            The model, with the mean headings `heading_a` and `heading_b` of the two species.

    Examples:
        >>> from skcomplex.models.physics.active_matter import nonreciprocal_vicsek
        >>> nonreciprocal_vicsek(20).elements[0].size
        20
    """
    rng = np.random.default_rng(random_state)
    species = (np.arange(n_movers) % 2).astype(float)
    movers = NonreciprocalMover.group(
        'movers',
        n_movers,
        x=rng.uniform(0, size, (n_movers, 2)),
        theta=rng.uniform(-math.pi, math.pi, n_movers),
        species=species,
        J_ab=J_ab,
        J_ba=J_ba,
        eta=eta,
    )
    return System(components=[movers], space=ContinuousSpace(size, size, boundary='periodic'), time=DiscreteTime())
