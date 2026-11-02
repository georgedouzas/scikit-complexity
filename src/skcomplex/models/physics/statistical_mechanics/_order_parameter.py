"""Order parameter of Landau theory relaxing down its free energy, `dphi/dt = -Gamma dF/dphi`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class OrderParameterElement(Element):
    """Order parameter of Landau theory relaxing down its free energy, `dphi/dt = -Gamma dF/dphi`.

    The free energy is `F = a (T - Tc) phi^2 / 2 + b phi^4 / 4 - h phi`.

    Args:
        label:
            The label of the order parameter.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        phi:
            The initial order parameter.

        a:
            The coefficient of the quadratic term per unit temperature.

        b:
            The coefficient of the quartic term.

        T:
            The temperature.

        Tc:
            The critical temperature.

        h:
            The external field.

        Gamma:
            The kinetic coefficient.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import OrderParameterElement
        >>> OrderParameterElement('magnet').Tc
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phi: Annotated[float, StateSpec()] = 0.1,
        a: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        b: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        T: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        Tc: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        h: Annotated[float, ParameterSpec()] = 0.0,
        Gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phi = phi
        self.a = a
        self.b = b
        self.T = T
        self.Tc = Tc
        self.h = h
        self.Gamma = Gamma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the relaxation down the gradient of the free energy.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        force = s.a * (s.T - s.Tc) * s.phi + s.b * s.phi**3 - s.h
        return [RateLaw(s.phi, -s.Gamma * force)]
