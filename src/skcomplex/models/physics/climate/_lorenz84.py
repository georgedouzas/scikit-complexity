"""Atmosphere of Lorenz-84, the general circulation reduced to a westerly jet and a travelling wave of eddies."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class Lorenz84Element(Element):
    """Atmosphere of Lorenz-84, the general circulation reduced to a westerly jet and a travelling wave of eddies.

    The jet `x` and the cosine and sine phases `y` and `z` of the eddies follow `dx/dt = -y^2 - z^2 - a x + a F`,
    `dy/dt = x y - b x z - y + G`, and `dz/dt = b x y + x z - z`.

    Args:
        label:
            The label of the atmosphere.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial strength of the westerly jet.

        y:
            The initial cosine phase of the eddies.

        z:
            The initial sine phase of the eddies.

        a:
            The damping of the jet relative to the eddies.

        b:
            The displacement of the eddies by the jet.

        F:
            The forcing of the jet by the equator-to-pole temperature contrast.

        G:
            The forcing of the eddies by the land-sea contrast.

    Examples:
        >>> from skcomplex.models.physics.climate import Lorenz84Element
        >>> Lorenz84Element('atmosphere').F
        8.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        y: Annotated[float, StateSpec()] = 1.0,
        z: Annotated[float, StateSpec()] = 1.0,
        a: Annotated[float, ParameterSpec()] = 0.25,
        b: Annotated[float, ParameterSpec()] = 4.0,
        F: Annotated[float, ParameterSpec()] = 8.0,
        G: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.b = b
        self.F = F
        self.G = G

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the forcing and damping of the jet and its exchange with the eddies.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of the jet and of both phases of the eddies.
        """
        return [
            RateLaw(s.x, -(s.y**2) - s.z**2 - s.a * s.x + s.a * s.F),
            RateLaw(s.y, s.x * s.y - s.b * s.x * s.z - s.y + s.G),
            RateLaw(s.z, s.b * s.x * s.y + s.x * s.z - s.z),
        ]
