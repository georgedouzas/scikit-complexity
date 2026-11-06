"""Particle of a Grégoire-Chaté flock, which aligns, repels close neighbours, and attracts distant ones."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, atan2, cos, oo, pi, sin, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import neighbors, uniform, where
from ....core.components.spaces import RadiusNeighborhood
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

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
