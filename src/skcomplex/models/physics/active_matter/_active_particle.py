"""Active particle in the plane, whose position diffuses, `dx = sqrt(2 D_t) dW`, and whose orientation turns."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import oo, pi, sqrt
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, NoiseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec

POINTS = VisualSpec('points', {'position': 'x', 'radius': 0.2})


class ActiveParticleElement(Element):
    """Active particle in the plane, whose position diffuses, `dx = sqrt(2 D_t) dW`, and whose orientation turns.

    The orientation turns at the angular velocity `omega` and diffuses, `dtheta = omega dt + sqrt(2 D_r) dW`. A
    `SelfPropulsionInteraction` adds the drift `v (cos theta, sin theta)` of the position. With `omega = 0` it is an
    active Brownian particle, and otherwise a chiral active particle.

    Args:
        label:
            The label of the particle or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        theta:
            The initial orientation.

        D_t:
            The translational diffusion coefficient.

        D_r:
            The rotational diffusion coefficient.

        omega:
            The angular velocity of a chiral particle.

    Examples:
        >>> from skcomplex.models.physics.active_matter import ActiveParticleElement
        >>> ActiveParticleElement('swimmer', D_r=2.0).D_r
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = POINTS,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
        theta: Annotated[float, StateSpec(period=2 * pi, role='direction')] = 0.0,
        D_t: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        D_r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        omega: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.theta = theta
        self.D_t = D_t
        self.D_r = D_r
        self.omega = omega

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the translational noise, and the turning and the rotational noise.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            NoiseLaw(s.x, sqrt(2 * s.D_t)),
            RateLaw(s.theta, s.omega),
            NoiseLaw(s.theta, sqrt(2 * s.D_r)),
        ]
