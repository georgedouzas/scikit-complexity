"""Point of the inverse iteration of `z -> z² + omega`, which moves to a square root of `z - omega` drawn at random."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import CC, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import bernoulli
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class InverseJuliaPointElement(Element):
    """Point of the inverse iteration of `z -> z² + omega`, which moves to a square root of `z - omega` drawn at random.

    The point moves to `+√(z - omega)` with probability `p` and to `-√(z - omega)` otherwise, with the principal
    square root.

    Args:
        label:
            The label of the element or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        z:
            The initial point.

        omega:
            The constant of the quadratic map.

        p:
            The probability of the positive square root.

    Examples:
        >>> from skcomplex.systems.mathematics.fractals import InverseJuliaPointElement
        >>> InverseJuliaPointElement('point', p=0.3).p
        0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        z: Annotated[object, StateSpec(domain=CC)] = 0.5,
        omega: Annotated[object, ParameterSpec(domain=CC)] = 0.0,
        p: Annotated[object, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.z = z
        self.omega = omega
        self.p = p

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the move to the square root drawn in the step.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [UpdateLaw(s.z, (2 * bernoulli('branch', s.p) - 1) * sqrt(s.z - s.omega))]
