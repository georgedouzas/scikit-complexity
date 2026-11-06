"""Site of the complex Ginzburg-Landau equation `A_t = A + (1 + i b) laplacian(A) - (1 - i c) |A|^2 A`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import laplacian
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class GinzburgLandauElement(Element):
    """Site of the complex Ginzburg-Landau equation `A_t = A + (1 + i b) laplacian(A) - (1 - i c) |A|^2 A`.

    The amplitude `A = u + i v` is held as its real and imaginary parts. Without the Laplacian every site is a
    Stuart-Landau oscillator, and the dispersion `b` and the nonlinear frequency shift `c` set whether plane waves are
    stable or break into spiral defect chaos.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        u:
            The initial real part of the amplitude.

        v:
            The initial imaginary part of the amplitude.

        b:
            The linear dispersion.

        c:
            The nonlinear frequency shift.

    Attributes:
        default_visual_:
            The visual of the site when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation import GinzburgLandauElement
        >>> GinzburgLandauElement('site').c
        -1.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        v: Annotated[float, StateSpec()] = 0.0,
        b: Annotated[float, ParameterSpec()] = 1.0,
        c: Annotated[float, ParameterSpec()] = -1.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.v = v
        self.b = b
        self.c = c

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the real and imaginary parts of the amplitude.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rates of `u` and `v`.
        """
        s = namespace
        square = s.u**2 + s.v**2
        return [
            RateLaw(s.u, s.u + laplacian(s.u) - s.b * laplacian(s.v) - square * (s.u - s.c * s.v)),
            RateLaw(s.v, s.v + s.b * laplacian(s.u) + laplacian(s.v) - square * (s.c * s.u + s.v)),
        ]
