"""Mechatronic models: a DC motor from components and a Lagrangian, and a quarter-car suspension."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ....core.interactions import ConnectionInteraction, EulerLagrangeInteraction
from ....core.laws.utils import derivative, step
from ....core.system import System
from ....specs import ObservableSpec
from ..._catalog import register_model
from ..common.electrical import GroundElement, InductorElement, ResistorElement, VoltageSourceElement
from ..common.mechanics import SpringDamperInteraction
from ._motor import ArmatureElement, InertiaElement, MotorCouplingInteraction
from ._quarter_car import QuarterCarElement


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


@register_model(
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
            The model, with the angle `rotor.phi`, the angular velocity `rotor.phi_d1`, and the current `winding.p_i`.

    Examples:
        >>> from skcomplex.models.engineering.mechatronics import dc_motor
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


@register_model(
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
            The model, with the displacements `car.y1` of the body and `car.y2` of the wheel.

    Examples:
        >>> from skcomplex.models.engineering.mechatronics import quarter_car
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
