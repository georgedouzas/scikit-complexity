"""Electromagnetic vibrating table: a shaker.

A supplied electromagnet pulls its table against a spring.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, oo, sin, tan
from sage.sets.real_set import RealSet

from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


class VibratingTableSystem(DynamicalSystem):
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
        >>> VibratingTableSystem(linear=True).get_params()['linear']
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
        super().__init__(label=label)
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

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'shaker'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'i2': DeclaredVariableSpec(0.0, StateSpec()),
            'uC': DeclaredVariableSpec(0.0, StateSpec()),
            'x': DeclaredVariableSpec(0.0, StateSpec()),
            'v': DeclaredVariableSpec(0.0, StateSpec()),
            'U': DeclaredVariableSpec(self.U, ParameterSpec()),
            'omega': DeclaredVariableSpec(self.omega, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'R': DeclaredVariableSpec(self.R, ParameterSpec(domain=RealSet.open(0, oo))),
            'L': DeclaredVariableSpec(self.L, ParameterSpec(domain=RealSet.open(0, oo))),
            'C': DeclaredVariableSpec(self.C, ParameterSpec(domain=RealSet.open(0, oo))),
            'K': DeclaredVariableSpec(self.K, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'm': DeclaredVariableSpec(self.m, ParameterSpec(domain=RealSet.open(0, oo))),
            'B': DeclaredVariableSpec(self.B, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'k1': DeclaredVariableSpec(self.k1, ParameterSpec()),
            'k2': DeclaredVariableSpec(self.k2, ParameterSpec()),
            'alpha': DeclaredVariableSpec(self.alpha, ParameterSpec(domain=RealSet.open(0, oo))),
        }

    def _get_equations(self) -> Equations:
        """Return the rates of the current, the capacitor voltage, and the table motion."""
        return {
            'i2': lambda s: (s.target.uC - s.target.k1 * s.target.v) / s.target.L,
            'uC': lambda s: (self._supply(s) - s.target.uC) / (s.target.R * s.target.C) - s.target.i2 / s.target.C,
            'x': lambda s: s.target.v,
            'v': lambda s: (self._get_force(s) - s.target.K * s.target.x - s.target.B * s.target.v) / s.target.m,
        }

    def _supply(self, s: SimpleNamespace) -> Expression:
        """Return `supply` from the namespace of an equation."""
        return s.target.U * sin(s.target.omega * s.t)

    def _get_force(self, s: SimpleNamespace) -> Expression:
        """Return the force of the electromagnet.

        It is `k2 tan(alpha i2) / alpha`, or its linearization `k2 i2`.
        """
        force: Expression = (
            s.target.k2 * s.target.i2
            if self.linear
            else s.target.k2 * tan(s.target.alpha * s.target.i2) / s.target.alpha
        )
        return force
