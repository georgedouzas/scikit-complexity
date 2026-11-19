"""Nonreciprocal flocking.

Two species of Vicsek movers align with each other differently.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, I, exp, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.laws.utils import neighbors, uniform
from ....core.components.spaces import BaseSpace, ContinuousSpace, RadiusNeighborhood
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.particles import PropulsionStepInteraction

SPECIES = VisualSpec('points', {'position': 'x', 'radius': 0.04, 'color_by': 'species', 'color_range': (0.0, 1.0)})


class NonreciprocalMoverElement(Element):
    """Movers of species A (0) or B (1) with the positions `x` and the headings `theta`.

    Args:
        label:
            The label of the movers.

        x:
            The positions, one row per mover or shared.

        theta:
            The headings, one per mover or shared.

        species:
            The species, 0 for A or 1 for B, one per mover or shared.

        size:
            The number of movers, or `None` for one.

        space:
            The continuous space of the movers, or `None` for none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import NonreciprocalMoverElement
        >>> NonreciprocalMoverElement('mover', species=1.0).species
        1.0
    """

    visual = SPECIES

    def __init__(
        self,
        label: str,
        *,
        x: object = (0.0, 0.0),
        theta: object = 0.0,
        species: object = 0.0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.x = x
        self.theta = theta
        self.species = species


class NonreciprocalAlignmentInteraction(Interaction):
    """Vicsek alignment of two species whose couplings to each other differ.

    Every step the heading becomes the direction of `sum_j J(species_i, species_j) exp(i theta_j)` over the movers
    within `radius`, itself included, plus uniform noise of width `eta`. `J_aa` and `J_bb` align a species with itself,
    and `J_ab` and `J_ba` align A with B and B with A, so `J_ab != J_ba` makes the alignment nonreciprocal. It is a
    pair of bird species, or two kinds of robots, that chase each other.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the movers, with the positions `x`, the headings `theta`, and the species `species`.

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
        >>> from skcomplex.systems.physics.active_matter import NonreciprocalAlignmentInteraction
        >>> NonreciprocalAlignmentInteraction('align', target='movers', J_ba=-1.0).J_ba
        -1.0
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        eta: Annotated[float, RealSet.closed_open(0, oo)] = 0.3,
        J_aa: float = 1.0,
        J_ab: float = 1.0,
        J_ba: float = 1.0,
        J_bb: float = 1.0,
        radius: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.eta = eta
        self.J_aa = J_aa
        self.J_ab = J_ab
        self.J_ba = J_ba
        self.J_bb = J_bb
        self.radius = radius

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the heading alignment of the target with its species-weighted neighbours, plus noise.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            first, second = s.target.species, s.target.other.species
            coupling = (
                s.J_aa * (1 - first) * (1 - second)
                + s.J_ab * (1 - first) * second
                + s.J_ba * first * (1 - second)
                + s.J_bb * first * second
            )
            over = RadiusNeighborhood(self.radius, center=True)
            direction = neighbors(s.target.other.theta, over=over, reduce='circular-mean', weight=coupling)
            self.laws_ = [UpdateLaw(s.target.theta, direction + uniform('noise', -s.eta / 2, s.eta / 2))]


def _get_heading_a(member: SimpleNamespace) -> Expression:
    """Return the heading of a member of species A, doubled.

    The doubling makes the mean over both species the mean over A.
    """
    return 2 * (1 - member.species) * exp(I * member.theta)


def _get_heading_b(member: SimpleNamespace) -> Expression:
    """Return the heading of a member of species B, doubled.

    The doubling makes the mean over both species the mean over B.
    """
    return 2 * member.species * exp(I * member.theta)


class NonreciprocalVicsekSystem(BaseSystem):
    """Two species of Vicsek movers in a periodic box with nonreciprocal alignment.

    The alignment of each species with the other differs between the two.

    Half the movers are species A and half species B, each aligning with its own species. With `J_ab = J_ba` the two
    species flock together. With `J_ab = -J_ba`, A aligns with B while B antialigns with A, so neither settles: the mean
    headings of the species lock at a quarter turn from each other and rotate together, the chiral phase of Fruchart et
    al. (2021), which breaks the time-reversal symmetry that reciprocal alignment keeps. The system has the mean
    headings `heading_a` and `heading_b` of the two species.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import NonreciprocalVicsekSystem
        >>> NonreciprocalVicsekSystem(20).prepare().system_.elements[0].size
        20
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'flocking', 'nonreciprocal', 'vicsek', 'chiral phase')
    references = ('Fruchart, Hanai, Littlewood, and Vitelli 2021', 'te Vrugt and Wittkowski 2024')
    until = 300.0
    dt = 1.0
    observables = (
        Observable('heading_a', _get_heading_a, group='movers'),
        Observable('heading_b', _get_heading_b, group='movers'),
    )

    def __init__(
        self,
        n_movers: int = 400,
        *,
        size: float = 5.0,
        J_ab: float = 1.0,
        J_ba: float = -1.0,
        eta: float = 0.3,
        random_state: int | np.random.Generator | None = 1,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_movers = n_movers
        self.size = size
        self.J_ab = J_ab
        self.J_ba = J_ba
        self.eta = eta
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of two species of Vicsek movers in a periodic box, aligning nonreciprocally.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        movers = NonreciprocalMoverElement(
            'movers',
            x=rng.uniform(0, self.size, (self.n_movers, 2)),
            theta=rng.uniform(-math.pi, math.pi, self.n_movers),
            species=(np.arange(self.n_movers) % 2).astype(float),
            size=self.n_movers,
            space=ContinuousSpace(extent=(self.size, self.size), boundary='periodic'),
        )
        alignment = NonreciprocalAlignmentInteraction(
            'align',
            target='movers',
            eta=self.eta,
            J_ab=self.J_ab,
            J_ba=self.J_ba,
        )
        self.system_ = System(
            self.label,
            elements=[movers],
            interactions=[alignment, PropulsionStepInteraction('propulsion', target='movers', speed=0.05)],
        )
