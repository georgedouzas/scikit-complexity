"""Mechatronic systems: a DC motor, a quarter-car suspension, a levitated ball, and a vibrating table."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ....core.components.interactions import ConnectionInteraction, EulerLagrangeInteraction, ImpulseInteraction
from ....core.components.laws.utils import derivative, step
from ....core.components.system import System
from ....core.specs import ObservableSpec
from ..._catalog import register_system
from ..common.electrical import GroundElement, InductorElement, ResistorElement, VoltageSourceElement
from ..common.mechanics import SpringDamperInteraction
from ._levitated_ball import LevitatedBallElement
from ._motor import ArmatureElement, InertiaElement, MotorCouplingInteraction
from ._quarter_car import QuarterCarElement
from ._vibrating_table import VibratingTableElement

IMPULSE_PERIOD = 4.0


def _speed_of_rotor(s: SimpleNamespace) -> Expression:
    """Return the angular velocity of the rotor."""
    speed: Expression = s.rotor.phi_d1
    return speed


def _current_of_winding(s: SimpleNamespace) -> Expression:
    """Return the current of the armature."""
    current: Expression = s.winding.p_i
    return current


def _rotor_lagrangian(s: SimpleNamespace) -> Expression:
    """Return the kinetic energy of the rotor."""
    energy: Expression = s.target.J * derivative(s.target.phi, 1) ** 2 / 2
    return energy


def _rotor_dissipation(s: SimpleNamespace) -> Expression:
    """Return the Rayleigh dissipation of the viscous friction of the rotor."""
    dissipation: Expression = s.target.b * derivative(s.target.phi, 1) ** 2 / 2
    return dissipation


def _spring_energy(s: SimpleNamespace) -> Expression:
    """Return the energy of the suspension spring and of the tire on the road step."""
    road = step(s.t, s.car.at, s.car.height)
    energy: Expression = s.car.k * (s.car.y1 - s.car.y2) ** 2 / 2 + s.car.kt * (s.car.y2 - road) ** 2 / 2
    return energy


@register_system(
    field='engineering/mechatronics',
    tags=('mechatronics', 'circuit', 'acausal components', 'euler-lagrange', 'ode'),
    references=('Fritzson 2015',),
    defaults={'until': 10.0, 'dt': 0.01},
    observables=lambda system: [
        ObservableSpec('speed', _speed_of_rotor),
        ObservableSpec('current', _current_of_winding),
    ],
)
def dc_motor(
    V: float = 12.0,
    *,
    R: float = 1.0,
    L: float = 0.5,
    k: float = 0.5,
    J: float = 0.1,
    b: float = 0.2,
) -> System:
    """Return a DC motor whose armature circuit of components drives an inertia on a Lagrangian coordinate.

    A voltage source drives a resistor, an inductor, and the armature in series. The armature has the voltage `k phi'`,
    and its current exerts the torque `k i` on the angle of an inertia with viscous friction, so the angular velocity
    settles at `k V / (R b + k^2)`.

    Args:
        V:
            The voltage of the source.

        R:
            The resistance of the armature circuit.

        L:
            The inductance of the armature circuit.

        k:
            The motor constant.

        J:
            The moment of inertia of the rotor.

        b:
            The coefficient of viscous friction of the rotor.

    Returns:
        system:
            The system, with the angle `rotor.phi`, the angular velocity `rotor.phi_d1`, and the current `winding.p_i`.

    Examples:
        >>> from skcomplex.systems.engineering.mechatronics import dc_motor
        >>> dc_motor(k=0.3).interactions[-1].k
        0.3
    """
    return System(
        elements=[
            VoltageSourceElement('source', V=V),
            ResistorElement('resistor', R=R),
            InductorElement('coil', L=L),
            ArmatureElement('winding'),
            GroundElement('ground'),
            InertiaElement('rotor', J=J, b=b),
        ],
        interactions=[
            ConnectionInteraction('supply', members=['source', 'resistor'], ports=['p', 'p']),
            ConnectionInteraction('series', members=['resistor', 'coil'], ports=['n', 'p']),
            ConnectionInteraction('terminal', members=['coil', 'winding'], ports=['n', 'p']),
            ConnectionInteraction('neutral', members=['winding', 'source', 'ground'], ports=['n', 'n', 'p']),
            EulerLagrangeInteraction(
                'rotation',
                target='rotor',
                states='phi',
                lagrangian=_rotor_lagrangian,
                dissipation=_rotor_dissipation,
            ),
            MotorCouplingInteraction('motor', armature='winding', inertia='rotor', k=k),
        ],
    )


@register_system(
    field='engineering/mechatronics',
    tags=('mechatronics', 'suspension', 'mass spring damper', 'ode', 'second order'),
    references=('Fritzson 2015',),
    defaults={'until': 10.0, 'dt': 0.005},
    observables=lambda system: [ObservableSpec('energy', _spring_energy)],
)
def quarter_car(
    height: float = 0.05,
    *,
    at: float = 0.5,
    m1: float = 250.0,
    m2: float = 30.0,
    k: float = 16000.0,
    b: float = 1000.0,
    kt: float = 160000.0,
) -> System:
    """Return a quarter-car suspension driving over a road step.

    A body of mass `m1` rests on a spring and a damper above a wheel of mass `m2`, and a tire spring joins the wheel to
    the road, which steps to `height` at time `at`. Both displacements settle at the height of the step.

    Args:
        height:
            The height of the road step.

        at:
            The time of the road step.

        m1:
            The mass of the body.

        m2:
            The mass of the wheel.

        k:
            The stiffness of the suspension.

        b:
            The damping of the suspension.

        kt:
            The stiffness of the tire.

    Returns:
        system:
            The system, with the displacements `car.y1` of the body and `car.y2` of the wheel.

    Examples:
        >>> from skcomplex.systems.engineering.mechatronics import quarter_car
        >>> quarter_car(height=0.1).elements[0].height
        0.1
    """
    car = QuarterCarElement('car', m1=m1, m2=m2, k=k, b=b, kt=kt, height=height, at=at)
    suspension = SpringDamperInteraction(
        'suspension',
        target='car',
        states=('y1', 'y2'),
        stiffness='k',
        damping='b',
        divisor={'y1': 'm1', 'y2': 'm2'},
    )
    return System(elements=[car], interactions=[suspension])


@register_system(
    field='engineering/mechatronics',
    tags=('mechatronics', 'levitation', 'impulses', 'events', 'ode'),
    references=('Pilipchuk 2010', 'Trzaska 2023'),
    defaults={'until': 20.0, 'dt': 0.01},
)
def magnetic_levitation(v: float | None = None, *, b: float = 1.0, m: float = 0.2) -> System:
    """Return a conductive ball levitated in a vacuum by a field that pushes and pulls it with alternating impulses.

    The velocity follows `v' + b v = m Σ_k [δ(t + 1 - 4 k) - δ(t - 1 - 4 k)]`: it relaxes freely, rises by `m` at the
    times `4 k - 1`, and falls by `m` at the times `4 k + 1`. Its periodic solution, with the triangular wave
    `τ(t) = (2 / π) arcsin(sin(π t / 2))` and the rectangular wave `e(t) = sign(cos(π t / 2))`, is
    `v = m / (2 cosh b) exp(-b τ(t) e(t)) e(t)`, which the ball follows from the start when it starts at
    `m / (2 cosh b)`.

    Args:
        v:
            The initial velocity, or `None` for the value `m / (2 cosh b)` of the periodic solution at time 0.

        b:
            The damping rate.

        m:
            The strength of every impulse.

    Returns:
        system:
            The system, with the velocity `ball.v`, raised by the impulses `push` and lowered by the impulses `pull`.

    Examples:
        >>> from skcomplex.systems.engineering.mechatronics import magnetic_levitation
        >>> round(magnetic_levitation(b=1.0, m=0.2).elements[0].v, 4)
        0.0648
    """
    start = m / (2 * math.cosh(b)) if v is None else v
    push = {'v': lambda s: s.target.v + s.target.m}
    pull = {'v': lambda s: s.target.v - s.target.m}
    return System(
        elements=[LevitatedBallElement('ball', v=start, b=b, m=m)],
        interactions=[
            ImpulseInteraction('push', target='ball', reset=push, period=IMPULSE_PERIOD, start=3.0),
            ImpulseInteraction('pull', target='ball', reset=pull, period=IMPULSE_PERIOD, start=1.0),
        ],
    )


@register_system(
    field='engineering/mechatronics',
    tags=('mechatronics', 'electromagnet', 'vibration', 'forcing', 'ode'),
    references=('Trzaska 2023',),
    defaults={'until': 40.0, 'dt': 0.01},
)
def vibrating_table(
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
) -> System:
    """Return an electromagnetic vibrating table, a shaker driven by a sinusoidal supply after its switch closes.

    The source `u = U sin(omega t)` feeds a resistor and a capacitor, whose voltage drives the current `i2` of the coil
    of an electromagnet against the back electromotive force `k1 v` of the table. The electromagnet pulls the table,
    held by a spring and a damper, with the force `g(i2) = k2 tan(alpha i2) / alpha`, whose slope `k2` at zero current
    is the one of Trzaska's linearized model. From rest, a transient of about 8 s dies out and the table vibrates
    steadily at the supply frequency, and the linearized model, `g(i2) = k2 i2`, gives nearly the same response since
    the current stays small.

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

    Returns:
        system:
            The system, with the current `shaker.i2`, the capacitor voltage `shaker.uC`, the displacement `shaker.x`,
            and the velocity `shaker.v`.

    Examples:
        >>> from skcomplex.systems.engineering.mechatronics import vibrating_table
        >>> vibrating_table(linear=True).elements[0].linear
        True
    """
    shaker = VibratingTableElement(
        'shaker',
        U=U,
        omega=omega,
        R=R,
        L=L,
        C=C,
        K=K,
        m=m,
        B=B,
        k1=k1,
        k2=k2,
        alpha=alpha,
        linear=linear,
    )
    return System(elements=[shaker])
