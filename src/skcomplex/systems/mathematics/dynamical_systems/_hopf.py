"""Normal form of the Hopf bifurcation in Cartesian coordinates, with the radial law `r' = mu r + a r^3`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class HopfElement(Element):
    """Normal form of the Hopf bifurcation in Cartesian coordinates, with the radial law `r' = mu r + a r^3`.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial real part.

        y:
            The initial imaginary part.

        mu:
            The bifurcation parameter.

        omega:
            The angular frequency.

        a:
            The cubic coefficient, negative for the supercritical bifurcation.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import HopfElement
        >>> HopfElement('form', mu=0.5).mu
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.1,
        y: Annotated[object, StateSpec()] = 0.0,
        mu: Annotated[object, ParameterSpec()] = 0.25,
        omega: Annotated[object, ParameterSpec()] = 1.0,
        a: Annotated[object, ParameterSpec()] = -1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.mu = mu
        self.omega = omega
        self.a = a

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the real and the imaginary part.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [
            RateLaw(s.x, s.mu * s.x - s.omega * s.y + s.a * s.x * (s.x**2 + s.y**2)),
            RateLaw(s.y, s.omega * s.x + s.mu * s.y + s.a * s.y * (s.x**2 + s.y**2)),
        ]
