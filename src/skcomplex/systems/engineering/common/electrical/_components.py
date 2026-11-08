"""Electrical components with ports and equations.

Connections assemble them into circuits.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import exp, oo
from sage.sets.real_set import RealSet

from .....core.components.elements import Element
from .....core.components.laws import BaseLaw, EquationLaw
from .....core.components.laws.utils import derivative
from .....core.specs import ConnectorSpec, ParameterSpec, StateSpec, VisualSpec

ELECTRICAL = ConnectorSpec('electrical', potential='v', flow='i')
POSITIVE = RealSet.open(0, oo)


class TwoPinElement(Element):
    """Component with a positive pin `p` and a negative pin `n`.

    Its currents balance. A subclass returns the equations of its branch from `_get_branch_laws`, in the voltage
    `p_v - n_v` and the current `p_i` from `p` to `n`.

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
    """

    law_classes = (EquationLaw,)

    ports: ClassVar[Mapping[str, ConnectorSpec]] = {'p': ELECTRICAL, 'n': ELECTRICAL}

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        p_v: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        p_i: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        n_v: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        n_i: Annotated[object, StateSpec(algebraic=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.p_v = p_v
        self.p_i = p_i
        self.n_v = n_v
        self.n_i = n_i

    def _get_branch_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the equations of the branch.

        There are none for an element that a coupling gives them.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The equation laws of the branch.
        """
        del s
        return []

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the balance of the currents of the pins and the equations of the branch."""
        return [EquationLaw(s.p_i + s.n_i, 0), *self._get_branch_laws(s)]


class ResistorElement(TwoPinElement):
    """Resistor that obeys Ohm's law, `p_v - n_v = R p_i`.

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

        R:
            The resistance.

    Examples:
        >>> from skcomplex.systems.engineering.common.electrical import ResistorElement
        >>> ResistorElement('load', R=2.0).R
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        p_v: object = 0.0,
        p_i: object = 0.0,
        n_v: object = 0.0,
        n_i: object = 0.0,
        R: Annotated[object, ParameterSpec(domain=POSITIVE)] = 1.0,
    ) -> None:
        super().__init__(label, visual, p_v=p_v, p_i=p_i, n_v=n_v, n_i=n_i)
        self.R = R

    def _get_branch_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return Ohm's law.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The equation law of the branch.
        """
        return [EquationLaw(s.p_v - s.n_v, s.R * s.p_i)]


class CapacitorElement(TwoPinElement):
    """Capacitor of voltage `u`, with `C u' = p_i` and `u = p_v - n_v`.

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

        u:
            The initial voltage.

        C:
            The capacitance.

    Examples:
        >>> from skcomplex.systems.engineering.common.electrical import CapacitorElement
        >>> CapacitorElement('tank', u=1.0).u
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        p_v: object = 0.0,
        p_i: object = 0.0,
        n_v: object = 0.0,
        n_i: object = 0.0,
        u: Annotated[object, StateSpec()] = 0.0,
        C: Annotated[object, ParameterSpec(domain=POSITIVE)] = 1.0,
    ) -> None:
        super().__init__(label, visual, p_v=p_v, p_i=p_i, n_v=n_v, n_i=n_i)
        self.u = u
        self.C = C

    def _get_branch_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the charge balance and the voltage of the capacitor.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The equation laws of the branch.
        """
        return [EquationLaw(s.C * derivative(s.u, 1), s.p_i), EquationLaw(s.u, s.p_v - s.n_v)]


class InductorElement(TwoPinElement):
    """Inductor of current `i`, with `L i' = p_v - n_v` and `i = p_i`.

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

        i:
            The initial current.

        L:
            The inductance.

    Examples:
        >>> from skcomplex.systems.engineering.common.electrical import InductorElement
        >>> InductorElement('coil', L=0.5).L
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        p_v: object = 0.0,
        p_i: object = 0.0,
        n_v: object = 0.0,
        n_i: object = 0.0,
        i: Annotated[object, StateSpec()] = 0.0,
        L: Annotated[object, ParameterSpec(domain=POSITIVE)] = 1.0,
    ) -> None:
        super().__init__(label, visual, p_v=p_v, p_i=p_i, n_v=n_v, n_i=n_i)
        self.i = i
        self.L = L

    def _get_branch_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the voltage and the current of the inductor.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The equation laws of the branch.
        """
        return [EquationLaw(s.L * derivative(s.i, 1), s.p_v - s.n_v), EquationLaw(s.i, s.p_i)]


class VoltageSourceElement(TwoPinElement):
    """Ideal source of the constant voltage `V = p_v - n_v`.

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

        V:
            The voltage.

    Examples:
        >>> from skcomplex.systems.engineering.common.electrical import VoltageSourceElement
        >>> VoltageSourceElement('battery', V=9.0).V
        9.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        p_v: object = 0.0,
        p_i: object = 0.0,
        n_v: object = 0.0,
        n_i: object = 0.0,
        V: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual, p_v=p_v, p_i=p_i, n_v=n_v, n_i=n_i)
        self.V = V

    def _get_branch_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the voltage of the source.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The equation law of the branch.
        """
        return [EquationLaw(s.p_v - s.n_v, s.V)]


class CurrentSourceElement(TwoPinElement):
    """Ideal source that drives a constant current from `n` through the source to `p`.

    Its current is `p_i = -current`.

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

        current:
            The current the source drives out of its positive pin.

    Examples:
        >>> from skcomplex.systems.engineering.common.electrical import CurrentSourceElement
        >>> CurrentSourceElement('source', current=0.5).current
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        p_v: object = 0.0,
        p_i: object = 0.0,
        n_v: object = 0.0,
        n_i: object = 0.0,
        current: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual, p_v=p_v, p_i=p_i, n_v=n_v, n_i=n_i)
        self.current = current

    def _get_branch_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the current of the source.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The equation law of the branch.
        """
        return [EquationLaw(s.p_i, -s.current)]


class GroundElement(Element):
    """Reference of the potentials of a circuit, whose one pin has zero potential.

    Args:
        label:
            The label of the element, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        p_v:
            The potential of the pin.

        p_i:
            The current into the pin.

    Examples:
        >>> from skcomplex.systems.engineering.common.electrical import GroundElement
        >>> sorted(GroundElement.ports)
        ['p']
    """

    law_classes = (EquationLaw,)

    ports: ClassVar[Mapping[str, ConnectorSpec]] = {'p': ELECTRICAL}

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        p_v: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        p_i: Annotated[object, StateSpec(algebraic=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.p_v = p_v
        self.p_i = p_i

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the zero potential of the pin."""
        return [EquationLaw(s.p_v, 0)]


class DiodeElement(TwoPinElement):
    """Exponential diode, with `p_i = Is (exp((p_v - n_v) / Vt) - 1)`.

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

        Is:
            The saturation current.

        Vt:
            The thermal voltage.

    Examples:
        >>> from skcomplex.systems.engineering.common.electrical import DiodeElement
        >>> DiodeElement('diode').Vt
        0.025
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        p_v: object = 0.0,
        p_i: object = 0.0,
        n_v: object = 0.0,
        n_i: object = 0.0,
        Is: Annotated[object, ParameterSpec(domain=POSITIVE)] = 1e-6,
        Vt: Annotated[object, ParameterSpec(domain=POSITIVE)] = 0.025,
    ) -> None:
        super().__init__(label, visual, p_v=p_v, p_i=p_i, n_v=n_v, n_i=n_i)
        self.Is = Is
        self.Vt = Vt

    def _get_branch_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the characteristic of the diode.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The equation law of the branch.
        """
        return [EquationLaw(s.p_i, s.Is * (exp((s.p_v - s.n_v) / s.Vt) - 1))]


class ChuaDiodeElement(TwoPinElement):
    """Nonlinear resistor of Chua's circuit.

    It has the current `m1 u + (m0 - m1) (|u + 1| - |u - 1|) / 2` at voltage `u`.

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

        m0:
            The slope of the characteristic inside the breakpoints.

        m1:
            The slope of the characteristic outside the breakpoints.

    Examples:
        >>> from skcomplex.systems.engineering.common.electrical import ChuaDiodeElement
        >>> round(ChuaDiodeElement('diode').m0, 3)
        -1.143
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        p_v: object = 0.0,
        p_i: object = 0.0,
        n_v: object = 0.0,
        n_i: object = 0.0,
        m0: Annotated[object, ParameterSpec()] = -8 / 7,
        m1: Annotated[object, ParameterSpec()] = -5 / 7,
    ) -> None:
        super().__init__(label, visual, p_v=p_v, p_i=p_i, n_v=n_v, n_i=n_i)
        self.m0 = m0
        self.m1 = m1

    def _get_branch_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the characteristic of the nonlinear resistor.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The equation law of the branch.
        """
        voltage = s.p_v - s.n_v
        return [EquationLaw(s.p_i, s.m1 * voltage + (s.m0 - s.m1) * (abs(voltage + 1) - abs(voltage - 1)) / 2)]


class OperationalAmplifierElement(Element):
    """Ideal operational amplifier.

    Its inputs draw no current and its output holds them at equal potentials. The output current returns through the
    supply of the amplifier, so no equation balances it.

    Args:
        label:
            The label of the element, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        p_v:
            The potential of the noninverting input.

        p_i:
            The current into the noninverting input.

        n_v:
            The potential of the inverting input.

        n_i:
            The current into the inverting input.

        out_v:
            The potential of the output.

        out_i:
            The current into the output.

    Examples:
        >>> from skcomplex.systems.engineering.common.electrical import OperationalAmplifierElement
        >>> sorted(OperationalAmplifierElement.ports)
        ['n', 'out', 'p']
    """

    ports: ClassVar[Mapping[str, ConnectorSpec]] = {'p': ELECTRICAL, 'n': ELECTRICAL, 'out': ELECTRICAL}

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        p_v: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        p_i: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        n_v: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        n_i: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        out_v: Annotated[object, StateSpec(algebraic=True)] = 0.0,
        out_i: Annotated[object, StateSpec(algebraic=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.p_v = p_v
        self.p_i = p_i
        self.n_v = n_v
        self.n_i = n_i
        self.out_v = out_v
        self.out_i = out_i

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the currents of the inputs and the equal potentials of the inputs."""
        return [EquationLaw(s.p_i, 0), EquationLaw(s.n_i, 0), EquationLaw(s.p_v, s.n_v)]
