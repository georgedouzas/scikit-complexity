"""Nonreciprocal flocking.

Two species of Vicsek movers align with each other differently.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, I, exp, oo, pi
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import neighbors, uniform
from ....core.components.spaces import ContinuousSpace, DiscreteTime, RadiusNeighborhood
from ....core.components.systems import BaseSystem, System
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..common.particles import PropulsionStepInteraction

SPECIES = VisualSpec("points", {"position": "x", "radius": 0.04, "color_by": "species", "color_range": (0.0, 1.0)})


def _heading_a(member: SimpleNamespace) -> Expression:
    """Return the heading of a member of species A, doubled.

    The doubling makes the mean over both species the mean over A.
    """
    return 2 * (1 - member.species) * exp(I * member.theta)


def _heading_b(member: SimpleNamespace) -> Expression:
    """Return the heading of a member of species B, doubled.

    The doubling makes the mean over both species the mean over B.
    """
    return 2 * member.species * exp(I * member.theta)


class NonreciprocalMoverElement(Element):
    """Vicsek mover of species A (0) or B (1).

    The mover aligns with the neighbours within `radius` by species couplings.

    Every step the heading becomes the direction of `sum_j J(species_i, species_j) exp(i theta_j)` over the movers
    within the radius, itself included, plus uniform noise of width `eta`. A propulsion step moves the mover along it.
    `J_aa` and `J_bb` align a species with itself, and `J_ab` and `J_ba` align A with B and B with A, so
    `J_ab != J_ba` makes the alignment nonreciprocal.

    Args:
        label:
            The label of the mover.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The position.

        theta:
            The heading.

        species:
            The species, 0 for A or 1 for B.

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
        >>> from skcomplex.systems.physics.active_matter import NonreciprocalMoverElement
        >>> NonreciprocalMoverElement('mover', J_ba=-1.0).J_ba
        -1.0
    """

    draws = True

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = SPECIES,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role="position")] = 0.0,
        theta: Annotated[float, StateSpec(period=2 * pi, role="direction")] = 0.0,
        species: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        eta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        J_aa: Annotated[float, ParameterSpec()] = 1.0,
        J_ab: Annotated[float, ParameterSpec()] = 1.0,
        J_ba: Annotated[float, ParameterSpec()] = 1.0,
        J_bb: Annotated[float, ParameterSpec()] = 1.0,
        radius: float = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.theta = theta
        self.species = species
        self.eta = eta
        self.J_aa = J_aa
        self.J_ab = J_ab
        self.J_ba = J_ba
        self.J_bb = J_bb
        self.radius = radius

    def _coupling(self, namespace: SimpleNamespace) -> Expression:
        """Return the alignment of a member with a neighbour, by the species of both."""
        s = namespace
        first, second = s.species, s.other.species
        return (
            s.J_aa * (1 - first) * (1 - second)
            + s.J_ab * (1 - first) * second
            + s.J_ba * first * (1 - second)
            + s.J_bb * first * second
        )

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the heading alignment with the species-weighted neighbours, plus noise.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The update of the heading.
        """
        s = namespace
        over = RadiusNeighborhood(self.radius, center=True)
        direction = neighbors(s.other.theta, over=over, reduce="circular-mean", weight=self._coupling(s))
        return [UpdateLaw(s.theta, direction + uniform("noise", -s.eta / 2, s.eta / 2))]


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

    Examples:
        >>> from skcomplex.systems.physics.active_matter import NonreciprocalVicsekSystem
        >>> NonreciprocalVicsekSystem(20).prepare().system_.elements[0].prepare().size_
        20
    """

    field = "physics/active_matter"
    topics = ("active matter", "flocking", "nonreciprocal", "vicsek", "chiral phase")
    references = ("Fruchart, Hanai, Littlewood, and Vitelli 2021", "te Vrugt and Wittkowski 2024")
    until = 300.0
    dt = 1.0
    observables = (
        ObservableSpec("heading_a", _heading_a, group="movers"),
        ObservableSpec("heading_b", _heading_b, group="movers"),
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
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.n_movers = n_movers
        self.size = size
        self.J_ab = J_ab
        self.J_ba = J_ba
        self.eta = eta
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return two species of Vicsek movers in a periodic box, aligning nonreciprocally.

        The alignment of each species with the other differs between the two.
        """
        rng = np.random.default_rng(self.random_state)
        species = (np.arange(self.n_movers) % 2).astype(float)
        movers = GroupElement(
            "movers",
            element=NonreciprocalMoverElement,
            size=self.n_movers,
            values={
                "x": rng.uniform(0, self.size, (self.n_movers, 2)),
                "theta": rng.uniform(-math.pi, math.pi, self.n_movers),
                "species": species,
                "J_ab": self.J_ab,
                "J_ba": self.J_ba,
                "eta": self.eta,
            },
        )
        return System(
            self.label,
            elements=[movers],
            interactions=[PropulsionStepInteraction("propulsion", target="movers", speed=0.05)],
            space=ContinuousSpace(extent=(self.size, self.size), boundary="periodic"),
            time=DiscreteTime(),
        )
