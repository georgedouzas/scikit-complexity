"""Cell that metabolizes a substance and exchanges it with its neighbours through gap junctions and nanotubes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import hill
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ExchangingCellElement(Element):
    """Cell that metabolizes a substance and exchanges it with its neighbours through gap junctions and nanotubes.

    The concentration follows `dx_i/dt = -alpha_i x_i / (beta_i + x_i) + Σ_j [g (x_j - x_i) + xi A_ij x_j]`, where the
    first term is Michaelis-Menten metabolism, the one law of the cell, `g` the exchange coefficient of the gap
    junctions, and `xi` the strength of the tunnelling nanotubes along the same links. The sum over the neighbours is a
    `CouplingInteraction` of a network of cells.

    Args:
        label:
            The label of the field of cells.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial concentration of the substance.

        alpha:
            The maximal rate of metabolism.

        beta:
            The half-saturation concentration of metabolism.

        g:
            The exchange coefficient of the gap junctions.

        xi:
            The strength of the nanotubes.

    Examples:
        >>> from skcomplex.systems.biology.physiology import ExchangingCellElement
        >>> ExchangingCellElement('cells').g
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        g: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        xi: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.alpha = alpha
        self.beta = beta
        self.g = g
        self.xi = xi

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the metabolism of the substance.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.x, -s.alpha * hill(s.x, s.beta))]
