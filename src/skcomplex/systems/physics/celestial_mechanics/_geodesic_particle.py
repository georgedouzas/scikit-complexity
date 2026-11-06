"""Test particle at the spacetime position `x = (t, r, theta, phi)` around a central mass, with its four-velocity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
import numpy.typing as npt

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec

R_INDEX = 1
PHI_INDEX = 3


def _equatorial_position(x: npt.NDArray[np.float64], **_: float) -> list[list[float]]:
    """Return the position `(r cos phi, r sin phi)` of a particle in the equatorial plane."""
    return [[x[R_INDEX] * np.cos(x[PHI_INDEX]), x[R_INDEX] * np.sin(x[PHI_INDEX])]]


def _central_mass(**_: float) -> list[list[float]]:
    """Return the position of the central mass at the origin."""
    return [[0.0, 0.0]]


class GeodesicParticleElement(Element):
    """Test particle at the spacetime position `x = (t, r, theta, phi)` around a central mass, with its four-velocity.

    The position is a vector state of order 2 in the Schwarzschild or Boyer-Lindquist coordinates of the space, and its
    companion `x_d1` is the four-velocity, the derivative of the position by the proper time of a massive particle or by
    the affine parameter of a light ray. The particle is drawn in the equatorial plane, at `(r cos phi, r sin phi)`.

    Args:
        label:
            The label of the particle.

        x:
            The initial position `(t, r, theta, phi)`.

        x_d1:
            The initial four-velocity.

        visual:
            How the particle is drawn, or `None` for its class visual.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import GeodesicParticleElement
        >>> GeodesicParticleElement('particle', x=[0.0, 10.0, 1.5708, 0.0]).x
        [0.0, 10.0, 1.5708, 0.0]
    """

    visual = VisualSpec('bodies', {'positions': _equatorial_position, 'trail': None, 'marks': _central_mass})

    def __init__(
        self,
        label: str,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, order=2)] = 0.0,
        x_d1: npt.ArrayLike = 0.0,
        visual: VisualSpec | None = None,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
