"""Thermal components with ports and equations, and the heating resistor that joins them to a circuit."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, EquationLaw
from ....core.components.laws.utils import derivative
from ....core.specs import ConnectorSpec, ParameterSpec, StateSpec, VisualSpec
from ..common.electrical import ELECTRICAL, TwoPinElement

THERMAL = ConnectorSpec('thermal', potential='T', flow='Q')
POSITIVE = RealSet.open(0, oo)


class HeatCapacityElement(Element):
    """Body of one temperature `T`, with `C T' = a_Q` and `T = a_T`.

    Args:
        label:
            The label of the element, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        a_T:
            The temperature of the port.

        a_Q:
            The heat flow into the port.

        T:
            The initial temperature of the body.

        C:
            The heat capacity.

    Examples:
        >>> from skcomplex.systems.engineering.control import HeatCapacityElement
        >>> HeatCapacityElement('room', T=15.0).T
        15.0
    """

    ports: ClassVar[Mapping[str, ConnectorSpec]] = {'a': THERMAL}

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a_T: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        a_Q: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        T: Annotated[object, StateSpec()] = 20.0,
        C: Annotated[object, ParameterSpec(domain=POSITIVE)] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a_T = a_T
        self.a_Q = a_Q
        self.T = T
        self.C = C

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the heat balance of the body and the temperature of its port."""
        return [EquationLaw(s.C * derivative(s.T, 1), s.a_Q), EquationLaw(s.T, s.a_T)]


class ThermalConductorElement(Element):
    """Conductor between two ports, with the heat flow `a_Q = G (a_T - b_T)` and `a_Q + b_Q = 0`.

    Args:
        label:
            The label of the element, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        a_T:
            The temperature of the first port.

        a_Q:
            The heat flow into the first port.

        b_T:
            The temperature of the second port.

        b_Q:
            The heat flow into the second port.

        G:
            The thermal conductance.

    Examples:
        >>> from skcomplex.systems.engineering.control import ThermalConductorElement
        >>> ThermalConductorElement('wall', G=2.0).G
        2.0
    """

    ports: ClassVar[Mapping[str, ConnectorSpec]] = {'a': THERMAL, 'b': THERMAL}

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a_T: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        a_Q: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        b_T: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        b_Q: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        G: Annotated[object, ParameterSpec(domain=POSITIVE)] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a_T = a_T
        self.a_Q = a_Q
        self.b_T = b_T
        self.b_Q = b_Q
        self.G = G

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the heat flow through the conductor and its balance."""
        return [EquationLaw(s.a_Q, s.G * (s.a_T - s.b_T)), EquationLaw(s.a_Q + s.b_Q, 0)]


class ConvectionElement(Element):
    """Surface that exchanges heat with a fluid of a fixed temperature, with `a_Q = h (a_T - fluid)`.

    Args:
        label:
            The label of the element, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        a_T:
            The temperature of the port.

        a_Q:
            The heat flow into the port.

        h:
            The convective conductance of the surface.

        fluid:
            The temperature of the fluid.

    Examples:
        >>> from skcomplex.systems.engineering.control import ConvectionElement
        >>> ConvectionElement('outside', fluid=5.0).fluid
        5.0
    """

    ports: ClassVar[Mapping[str, ConnectorSpec]] = {'a': THERMAL}

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a_T: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        a_Q: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        h: Annotated[object, ParameterSpec(domain=POSITIVE)] = 1.0,
        fluid: Annotated[object, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a_T = a_T
        self.a_Q = a_Q
        self.h = h
        self.fluid = fluid

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the heat flow into the fluid."""
        return [EquationLaw(s.a_Q, s.h * (s.a_T - s.fluid))]


class HeatingResistorElement(TwoPinElement):
    """Resistor that delivers its electric power to a thermal port `h` while its switch `on` is closed.

    The current is `on (p_v - n_v) / R`, and the heat flow into the thermal port is `-(p_v - n_v) p_i`.

    Args:
        label:
            The label of the element, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        p_v:
            The potential of the positive pin.

        p_i:
            The current into the positive pin.

        n_v:
            The potential of the negative pin.

        n_i:
            The current into the negative pin.

        h_T:
            The temperature of the thermal port.

        h_Q:
            The heat flow into the thermal port.

        on:
            The initial position of the switch, 1 for closed and 0 for open.

        R:
            The resistance.

    Examples:
        >>> from skcomplex.systems.engineering.control import HeatingResistorElement
        >>> sorted(HeatingResistorElement.ports)
        ['h', 'n', 'p']
    """

    ports: ClassVar[Mapping[str, ConnectorSpec]] = {'p': ELECTRICAL, 'n': ELECTRICAL, 'h': THERMAL}

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        p_v: object = 0.0,
        p_i: object = 0.0,
        n_v: object = 0.0,
        n_i: object = 0.0,
        h_T: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        h_Q: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        on: Annotated[object, StateSpec()] = 1.0,
        R: Annotated[object, ParameterSpec(domain=POSITIVE)] = 1.0,
    ) -> None:
        super().__init__(label, visual, p_v=p_v, p_i=p_i, n_v=n_v, n_i=n_i)
        self.h_T = h_T
        self.h_Q = h_Q
        self.on = on
        self.R = R

    def _get_branch_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the switched current of the resistor and the heat it delivers.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The equation laws of the branch.
        """
        voltage = s.p_v - s.n_v
        return [EquationLaw(s.R * s.p_i, s.on * voltage), EquationLaw(s.h_Q, -voltage * s.p_i)]
