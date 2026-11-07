"""Mechatronic systems: a DC motor from components and a Lagrangian, and a quarter-car suspension."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import ConnectionInteraction, EulerLagrangeInteraction, Interaction
from ....core.components.laws import BaseLaw, EquationLaw, RateLaw
from ....core.components.laws.utils import derivative, step
from ....core.components.system import System
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system
from ..common.electrical import GroundElement, InductorElement, ResistorElement, TwoPinElement, VoltageSourceElement
from ..common.mechanics import SpringDamperInteraction

POSITIVE = RealSet.open(0, oo)


class ArmatureElement(TwoPinElement):
    """Winding of a motor, a two-pin element whose voltage the coupling with the inertia of the motor gives.

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

    Examples:
        >>> from skcomplex.systems.engineering.mechatronics import ArmatureElement
        >>> sorted(ArmatureElement.ports)
        ['n', 'p']
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
    ) -> None:
        super().__init__(label, visual, p_v=p_v, p_i=p_i, n_v=n_v, n_i=n_i)


class InertiaElement(Element):
    """Rotating inertia of the angle `phi`, a coordinate of a Lagrangian.

    Args:
        label:
            The label of the element, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        phi:
            The initial angle.

        phi_d1:
            The initial angular velocity.

        J:
            The moment of inertia.

        b:
            The coefficient of viscous friction.

    Examples:
        >>> from skcomplex.systems.engineering.mechatronics import InertiaElement
        >>> InertiaElement('rotor', J=0.5).J
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phi: Annotated[object, StateSpec(order=2)] = 0.0,
        phi_d1: object = 0.0,
        J: Annotated[object, ParameterSpec(domain=POSITIVE)] = 1.0,
        b: Annotated[object, ParameterSpec(domain=POSITIVE)] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phi = phi
        self.phi_d1 = phi_d1
        self.J = J
        self.b = b


class QuarterCarElement(Element):
    """Body of displacement `y1` and wheel of displacement `y2`, with a tire between the wheel and a road step.

    The element holds the tire, whose force on the wheel is `kt (road - y2)`, and a suspension interaction joins the
    body and the wheel.

    Args:
        label:
            The label of the element, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        y1:
            The initial displacement of the body.

        y1_d1:
            The initial velocity of the body.

        y2:
            The initial displacement of the wheel.

        y2_d1:
            The initial velocity of the wheel.

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

        height:
            The height of the road step.

        at:
            The time of the road step.

    Examples:
        >>> from skcomplex.systems.engineering.mechatronics import QuarterCarElement
        >>> QuarterCarElement('car', height=0.05).height
        0.05
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        y1: Annotated[object, StateSpec(order=2)] = 0.0,
        y1_d1: object = 0.0,
        y2: Annotated[object, StateSpec(order=2)] = 0.0,
        y2_d1: object = 0.0,
        m1: Annotated[object, ParameterSpec(domain=POSITIVE)] = 250.0,
        m2: Annotated[object, ParameterSpec(domain=POSITIVE)] = 30.0,
        k: Annotated[object, ParameterSpec(domain=POSITIVE)] = 16000.0,
        b: Annotated[object, ParameterSpec(domain=POSITIVE)] = 1000.0,
        kt: Annotated[object, ParameterSpec(domain=POSITIVE)] = 160000.0,
        height: Annotated[object, ParameterSpec()] = 0.05,
        at: Annotated[object, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.y1 = y1
        self.y1_d1 = y1_d1
        self.y2 = y2
        self.y2_d1 = y2_d1
        self.m1 = m1
        self.m2 = m2
        self.k = k
        self.b = b
        self.kt = kt
        self.height = height
        self.at = at

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the acceleration of the wheel by the tire, the suspension apart."""
        return [RateLaw(s.y2, s.kt * (step(s.t, s.at, s.height) - s.y2) / s.m2)]


class MotorCouplingInteraction(Interaction):
    """Coupling of an armature and an inertia by a motor constant `k`.

    The voltage of the armature is the back electromotive force `k phi'`, and its current exerts the torque `k p_i` on
    the angle, which adds to the Euler-Lagrange equation of the angle.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        armature:
            The path of the armature.

        inertia:
            The path of the inertia.

        k:
            The motor constant.

    Examples:
        >>> from skcomplex.systems.engineering.mechatronics import MotorCouplingInteraction
        >>> MotorCouplingInteraction('motor', armature='winding', inertia='rotor', k=0.5).k
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        armature: Annotated[str, RoleSpec()],
        inertia: Annotated[str, RoleSpec()],
        k: Annotated[object, ParameterSpec(domain=POSITIVE)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.armature = armature
        self.inertia = inertia
        self.k = k

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the back electromotive force and the torque on the angle.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The equation laws of the coupling.
        """
        return [
            EquationLaw(s.armature.p_v - s.armature.n_v, s.k * derivative(s.inertia.phi, 1)),
            EquationLaw(-s.k * s.armature.p_i, 0, target=s.inertia.phi),
        ]


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
