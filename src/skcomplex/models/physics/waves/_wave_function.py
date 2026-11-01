"""Site of the wave function `psi = u + i w` of a particle with the potential `V`, by its real and imaginary parts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import laplacian
from ....specs import ParameterSpec, StateSpec, VisualSpec


class WaveFunctionElement(Element):
    """Site of the wave function `psi = u + i w` of a particle with the potential `V`, by its real and imaginary parts.

    The parts follow the Schrödinger equation `i psi' = -D laplacian(psi) + V psi`, so
    `u' = -D laplacian(w) + V w` and `w' = D laplacian(u) - V u`.

    Args:
        label:
            The label of the site.

        visual:
            How the site is drawn, or `None` for none.

        u:
            The initial real part.

        w:
            The initial imaginary part.

        D:
            The coefficient of the kinetic energy, `hbar / (2 m)`.

        V:
            The potential at the site.

    Examples:
        >>> from skcomplex.models.physics.waves import WaveFunctionElement
        >>> WaveFunctionElement('site', V=1.0).V
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        w: Annotated[float, StateSpec()] = 0.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        V: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.w = w
        self.D = D
        self.V = V

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the Schrödinger equation of the real and imaginary parts.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of the two parts.
        """
        return [
            RateLaw(s.u, -s.D * laplacian(s.w) + s.V * s.w),
            RateLaw(s.w, s.D * laplacian(s.u) - s.V * s.u),
        ]
