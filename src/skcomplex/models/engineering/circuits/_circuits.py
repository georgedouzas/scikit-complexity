"""Electric circuits."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import cos

from ....core.base import Domain, parameter, state
from ....core.model import Element, Equation, System, derivative
from ....core.runtime import register_model

Circuit = Element.define(
    'Circuit',
    states={'q': state(default=1.0, order=2)},
    parameters={
        'L': parameter(default=1.0, domain=Domain.POSITIVE),
        'R': parameter(default=0.1, domain=Domain.NONNEGATIVE),
        'C': parameter(default=1.0, domain=Domain.POSITIVE),
        'V': parameter(default=0.0),
        'omega': parameter(default=1.0, domain=Domain.NONNEGATIVE),
    },
    laws=lambda s: [
        Equation(s.L * derivative(s.q, 2) + s.R * derivative(s.q, 1) + s.q / s.C == s.V * cos(s.omega * s.t)),
    ],
)


@register_model(
    field='engineering/circuits',
    tags=('electromagnetism', 'circuit', 'ode', 'second order'),
    references=('Kirchhoff 1845',),
    defaults={'until': 50.0, 'dt': 0.05},
)
def rlc_circuit(
    q: float = 1.0,
    current: float = 0.0,
    *,
    L: float = 1.0,
    R: float = 0.1,
    C: float = 1.0,
    V: float = 0.0,
    omega: float = 1.0,
) -> System:
    """Return a series RLC circuit written as Kirchhoff's voltage law, `L q'' + R q' + q / C = V cos(ω t)`.

    Args:
        q:
            The initial charge of the capacitor.

        current:
            The initial current.

        L:
            The inductance.

        R:
            The resistance.

        C:
            The capacitance.

        V:
            The amplitude of the source voltage.

        omega:
            The angular frequency of the source.

    Returns:
        system:
            The model, with the circuit `circuit`, whose charge has order 2 and current `circuit.q_d1`.

    Examples:
        >>> from skcomplex.models.engineering.circuits import rlc_circuit
        >>> rlc_circuit(R=0.5).elements[0].R
        0.5
    """
    return System(elements=[Circuit('circuit', q=q, q_d1=current, L=L, R=R, C=C, V=V, omega=omega)])
