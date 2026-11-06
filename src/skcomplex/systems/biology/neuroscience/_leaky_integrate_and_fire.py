"""Leaky integrate-and-fire neuron, whose potential relaxes to `rest + resistance I` over the time `tau`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class LeakyIntegrateAndFireElement(Element):
    """Leaky integrate-and-fire neuron, whose potential relaxes to `rest + resistance I` over the time `tau`.

    The potential follows `tau v' = -(v - rest) + resistance I`. A spike reset fires the neuron. Units are mV, ms, MΩ,
    and nA.

    Args:
        label:
            The label of the neuron.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        v:
            The initial potential.

        rest:
            The resting potential.

        resistance:
            The membrane resistance.

        tau:
            The membrane time constant.

        I:
            The input current.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import LeakyIntegrateAndFireElement
        >>> LeakyIntegrateAndFireElement('neuron').tau
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        v: Annotated[float, StateSpec()] = -65.0,
        rest: Annotated[float, ParameterSpec()] = -65.0,
        resistance: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        I: Annotated[float, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.v = v
        self.rest = rest
        self.resistance = resistance
        self.tau = tau
        self.I = I

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the leaky integration of the input current.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.v, (-(s.v - s.rest) + s.resistance * s.I) / s.tau)]
