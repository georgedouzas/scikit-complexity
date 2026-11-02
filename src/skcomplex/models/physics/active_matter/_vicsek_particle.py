"""Particle of a Vicsek flock, whose heading becomes the mean heading of its neighbours plus noise every step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import oo, pi
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


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
        >>> from skcomplex.models.physics.active_matter import VicsekParticleElement
        >>> VicsekParticleElement('bird').eta
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, position=True, role='position')] = 0.0,
        theta: Annotated[float, StateSpec(period=2 * pi, role='direction')] = 0.0,
        eta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.theta = theta
        self.eta = eta
