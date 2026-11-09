"""Electromagnetic vibrating table: a shaker.

A supplied electromagnet pulls its table against a spring.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo, sin, tan
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class VibratingTableElement(Element):
    """Electromagnetic vibrating table, a shaker.

    A supplied electromagnet pulls its table against a spring. The source `u = U sin(omega t)` feeds a resistor `R` and
    a capacitor `C`, whose voltage `uC` drives the current `i2` of the coil `L` against the back electromotive force
    `k1 v` of the moving table. The electromagnet pulls the table of mass `m`, held by a spring `K` and a damper `B`,
    with the force `g(i2) = k2 tan(alpha i2) / alpha`, or with its linearization `k2 i2` at the origin. The states
    follow `i2' = (uC - k1 v) / L`, `uC' = (u - uC) / (R C) - i2 / C`, `x' = v`, and `m v' = g(i2) - K x - B v`.

    Args:
        label:
            The label of the shaker.

        visual:
            How the shaker is drawn, or `None` for `default_visual_`.

        i2:
            The initial current of the coil.

        uC:
            The initial voltage of the capacitor.

        x:
            The initial displacement of the table.

        v:
            The initial velocity of the table.

        U:
            The amplitude of the supply voltage.

        omega:
            The angular frequency of the supply voltage.

        R:
            The resistance.

        L:
            The inductance of the coil.

        C:
            The capacitance.

        K:
            The stiffness of the spring.

        m:
            The mass of the table.

        B:
            The damping coefficient.

        k1:
            The electromechanical constant of the back electromotive force.

        k2:
            The mechanical-electrical constant, the slope of the force of the electromagnet at zero current.

        alpha:
            The scale of the current in the characteristic of the electromagnet.

        linear:
            Whether the force of the electromagnet is its linearization `k2 i2`.

    Examples:
        >>> from skcomplex.systems.engineering.mechatronics import VibratingTableElement
        >>> VibratingTableElement('shaker').k2
        17.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        i2: Annotated[float, StateSpec()] = 0.0,
        uC: Annotated[float, StateSpec()] = 0.0,
        x: Annotated[float, StateSpec()] = 0.0,
        v: Annotated[float, StateSpec()] = 0.0,
        U: Annotated[float, ParameterSpec()] = 0.141,
        omega: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.5915,
        R: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.5,
        L: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.25,
        C: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        K: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 500.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        B: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        k1: Annotated[float, ParameterSpec()] = 0.5,
        k2: Annotated[float, ParameterSpec()] = 17.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.5,
        linear: bool = False,
    ) -> None:
        super().__init__(label, visual=visual)
        self.i2 = i2
        self.uC = uC
        self.x = x
        self.v = v
        self.U = U
        self.omega = omega
        self.R = R
        self.L = L
        self.C = C
        self.K = K
        self.m = m
        self.B = B
        self.k1 = k1
        self.k2 = k2
        self.alpha = alpha
        self.linear = linear

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the current, the capacitor voltage, and the table motion.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        supply = s.U * sin(s.omega * s.t)
        return [
            RateLaw(s.i2, (s.uC - s.k1 * s.v) / s.L),
            RateLaw(s.uC, (supply - s.uC) / (s.R * s.C) - s.i2 / s.C),
            RateLaw(s.x, s.v),
            RateLaw(s.v, (self._force(s) - s.K * s.x - s.B * s.v) / s.m),
        ]

    def _force(self, s: SimpleNamespace) -> Expression:
        """Return the force of the electromagnet.

        It is `k2 tan(alpha i2) / alpha`, or its linearization `k2 i2`.
        """
        force: Expression = s.k2 * s.i2 if self.linear else s.k2 * tan(s.alpha * s.i2) / s.alpha
        return force


class VibratingTableSystem(BaseSystem):
    """An electromagnetic vibrating table, a shaker driven by a sinusoidal supply.

    The supply drives it after its switch closes. The source `u = U sin(omega t)` feeds a resistor and a capacitor,
    whose voltage drives the current `i2` of the coil of an electromagnet against the back electromotive force `k1 v` of
    the table. The electromagnet pulls the table, held by a spring and a damper, with the force
    `g(i2) = k2 tan(alpha i2) / alpha`, whose slope `k2` at zero current is the one of Trzaska's linearized model. From
    rest, a transient of about 8 s dies out and the table vibrates steadily at the supply frequency, and the linearized
    model, `g(i2) = k2 i2`, gives nearly the same response since the current stays small.

    The system has the current `shaker.i2`, the capacitor voltage `shaker.uC`, the displacement `shaker.x`, and the
    velocity `shaker.v`.

    Args:
        U:
            The amplitude of the supply voltage in V.

        omega:
            The angular frequency of the supply in rad/s.

        R:
            The resistance in Ω.

        L:
            The inductance of the coil in H.

        C:
            The capacitance in F.

        K:
            The stiffness of the spring in N/m.

        m:
            The mass of the table in kg.

        B:
            The damping coefficient in Ns/m.

        k1:
            The electromechanical constant in Wb/m.

        k2:
            The mechanical-electrical constant, the slope of the force of the electromagnet at zero current.

        alpha:
            The scale of the current in the characteristic of the electromagnet in 1/A.

        linear:
            Whether the force of the electromagnet is its linearization `k2 i2`.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.mechatronics import VibratingTableSystem
        >>> VibratingTableSystem(linear=True).prepare().system_.elements[0].linear
        True
    """

    field = 'engineering/mechatronics'
    topics = ('mechatronics', 'electromagnet', 'vibration', 'forcing', 'ode')
    references = ('Trzaska 2023',)
    until = 40.0
    dt = 0.01

    def __init__(
        self,
        *,
        U: float = 0.141,
        omega: float = 1.5915,
        R: float = 1.5,
        L: float = 0.25,
        C: float = 0.1,
        K: float = 500.0,
        m: float = 10.0,
        B: float = 2.0,
        k1: float = 0.5,
        k2: float = 17.0,
        alpha: float = 2.5,
        linear: bool = False,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.U = U
        self.omega = omega
        self.R = R
        self.L = L
        self.C = C
        self.K = K
        self.m = m
        self.B = B
        self.k1 = k1
        self.k2 = k2
        self.alpha = alpha
        self.linear = linear

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        shaker = VibratingTableElement(
            'shaker',
            U=self.U,
            omega=self.omega,
            R=self.R,
            L=self.L,
            C=self.C,
            K=self.K,
            m=self.m,
            B=self.B,
            k1=self.k1,
            k2=self.k2,
            alpha=self.alpha,
            linear=self.linear,
        )
        return CompositeSystem(self.label, elements=[shaker])
