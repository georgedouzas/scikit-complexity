"""Particle of a delayed Vicsek flock, which keeps its past headings for the delayed or predicted alignment."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import oo, pi
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

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
