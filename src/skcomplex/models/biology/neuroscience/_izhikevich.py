"""Izhikevich neuron, a quadratic potential `v` and a linear recovery `u`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class IzhikevichElement(Element):
    """Izhikevich neuron, a quadratic potential `v` and a linear recovery `u`.

    The potential follows `v' = 0.04 v² + 5 v + 140 - u + I` and the recovery `u' = a (b v - u)`. A spike reset fires
    the neuron.

    Args:
        label:
            The label of the neuron.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        v:
            The initial potential.

        u:
            The initial recovery.

        a:
            The time scale of the recovery.

        b:
            The sensitivity of the recovery to the potential.

        I:
            The input current.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import IzhikevichElement
        >>> IzhikevichElement('neuron').b
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        v: Annotated[float, StateSpec()] = -65.0,
        u: Annotated[float, StateSpec()] = -13.0,
        a: Annotated[float, ParameterSpec()] = 0.02,
        b: Annotated[float, ParameterSpec()] = 0.2,
        I: Annotated[float, ParameterSpec()] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.v = v
        self.u = u
        self.a = a
        self.b = b
        self.I = I

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the quadratic potential and the linear recovery.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.v, 0.04 * s.v**2 + 5 * s.v + 140 - s.u + s.I), RateLaw(s.u, s.a * (s.b * s.v - s.u))]
