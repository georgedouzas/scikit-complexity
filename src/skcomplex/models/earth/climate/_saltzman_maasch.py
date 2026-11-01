"""Climate of the Saltzman-Maasch glacial cycles, the anomalies of ice, carbon dioxide, and deep ocean temperature."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos, pi

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class SaltzmanMaaschElement(Element):
    """Climate of the Saltzman-Maasch glacial cycles, the anomalies of ice, carbon dioxide, and deep ocean temperature.

    The anomalies of global ice volume `x`, atmospheric carbon dioxide `y`, and deep ocean temperature `z` follow
    `dx/dt = -x - y - u z - nu F(t)`, `dy/dt = -p z + r y - s y^2 - y^3`, and `dz/dt = -q (x + z)`, where
    `F(t) = cos(2π t0 t / 22.1) - cos(2π t0 t / 44.1)` is the orbital forcing of precession and obliquity.

    Args:
        label:
            The label of the climate.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial ice volume anomaly.

        y:
            The initial carbon dioxide anomaly.

        z:
            The initial deep ocean temperature anomaly.

        p:
            The effect of the deep ocean temperature on carbon dioxide.

        q:
            The relaxation rate of the deep ocean temperature to the ice volume.

        r:
            The growth rate of carbon dioxide by its own feedback.

        s:
            The quadratic saturation of carbon dioxide.

        u:
            The effect of the deep ocean temperature on ice volume.

        nu:
            The amplitude of the orbital forcing.

        t0:
            The time unit in thousands of years, which scales the orbital periods.

    Examples:
        >>> from skcomplex.models.earth.climate import SaltzmanMaaschElement
        >>> SaltzmanMaaschElement('climate').q
        2.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = -0.2,
        y: Annotated[float, StateSpec()] = 1.0,
        z: Annotated[float, StateSpec()] = 0.0,
        p: Annotated[float, ParameterSpec()] = 1.0,
        q: Annotated[float, ParameterSpec()] = 2.5,
        r: Annotated[float, ParameterSpec()] = 1.3,
        s: Annotated[float, ParameterSpec()] = 0.6,
        u: Annotated[float, ParameterSpec()] = 0.2,
        nu: Annotated[float, ParameterSpec()] = 0.0,
        t0: Annotated[float, ParameterSpec()] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.p = p
        self.q = q
        self.r = r
        self.s = s
        self.u = u
        self.nu = nu
        self.t0 = t0

    def _get_orbital_forcing(self, s: SimpleNamespace) -> Expression:
        """Return the quasiperiodic orbital forcing, with the periods of precession and obliquity.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            forcing:
                The orbital forcing.
        """
        return cos(2 * pi * s.t0 * s.t / 22.1) - cos(2 * pi * s.t0 * s.t / 44.1)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the feedbacks of ice volume, carbon dioxide, and deep ocean temperature, with orbital forcing.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of the three anomalies.
        """
        forcing = self._get_orbital_forcing(s)
        return [
            RateLaw(s.x, -s.x - s.y - s.u * s.z - s.nu * forcing),
            RateLaw(s.y, -s.p * s.z + s.r * s.y - s.s * s.y**2 - s.y**3),
            RateLaw(s.z, -s.q * (s.x + s.z)),
        ]
