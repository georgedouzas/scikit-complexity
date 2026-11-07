"""Functional unit of a tissue layer reduced to a phase, a cell oscillator `phi' = omega`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import pi

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class CellOscillatorElement(Element):
    """Functional unit of a tissue layer reduced to a phase, a cell oscillator `phi' = omega`.

    Healthy and pathological cells differ by their metabolism, which the natural frequency `omega` stands for.
    Couplings to other cells add to the rate of the phase.

    Args:
        label:
            The label of the cell or of the group of cells.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        phi:
            The initial phase.

        omega:
            The natural frequency.

    Examples:
        >>> from skcomplex.systems.biology.physiology import CellOscillatorElement
        >>> CellOscillatorElement('cell', omega=0.2).omega
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phi: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
        omega: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phi = phi
        self.omega = omega

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rotation of the phase.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.phi, s.omega)]
