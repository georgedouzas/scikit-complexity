"""Mechatronic systems: a DC motor and a quarter-car suspension.

The DC motor is built from components and a Lagrangian.
"""

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
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..common.electrical import GroundElement, InductorElement, ResistorElement, TwoPinElement, VoltageSourceElement
from ..common.mechanics import SpringDamperInteraction

POSITIVE = RealSet.open(0, oo)


class ArmatureElement(TwoPinElement):
    """Winding of a motor, a two-pin element.

    Its voltage is given by the coupling with the inertia of the motor.

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
    """Body of displacement `y1` and wheel of displacement `y2`.

    A tire lies between the wheel and a road step. The element holds the tire, whose force on the wheel is
    `kt (road - y2)`, and a suspension interaction joins the body and the wheel.

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

    law_classes = (RateLaw,)

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

    law_classes = (EquationLaw,)

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


def _get_speed_of_rotor(s: SimpleNamespace) -> Expression:
    """Return the angular velocity of the rotor."""
    speed: Expression = s.rotor.phi_d1
    return speed


def _get_current_of_winding(s: SimpleNamespace) -> Expression:
    """Return the current of the armature."""
    current: Expression = s.winding.p_i
    return current


class DcMotorSystem(BaseSystem):
    """A DC motor whose armature circuit drives an inertia.

    The armature circuit is made of components and the inertia is on a Lagrangian coordinate. A voltage source drives a
    resistor, an inductor, and the armature in series. The armature has the voltage `k phi'`, and its current exerts the
    torque `k i` on the angle of an inertia with viscous friction, so the angular velocity settles at
    `k V / (R b + k^2)`. The system has the angle `rotor.phi`, the angular velocity `rotor.phi_d1`, and the current
    `winding.p_i`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.mechatronics import DcMotorSystem
        >>> DcMotorSystem(k=0.3).prepare().system_.interactions[-1].k
        0.3
    """

    field = 'engineering/mechatronics'
    topics = ('mechatronics', 'circuit', 'acausal components', 'euler-lagrange', 'ode')
    references = ('Fritzson 2015',)
    until = 10.0
    dt = 0.01
    observables = (
        ObservableSpec('speed', _get_speed_of_rotor),
        ObservableSpec('current', _get_current_of_winding),
    )

    def __init__(
        self,
        V: float = 12.0,
        *,
        R: float = 1.0,
        L: float = 0.5,
        k: float = 0.5,
        J: float = 0.1,
        b: float = 0.2,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.V = V
        self.R = R
        self.L = L
        self.k = k
        self.J = J
        self.b = b

    @staticmethod
    def _get_rotor_lagrangian(s: SimpleNamespace) -> Expression:
        """Return the kinetic energy of the rotor."""
        energy: Expression = s.target.J * derivative(s.target.phi, 1) ** 2 / 2
        return energy

    @staticmethod
    def _get_rotor_dissipation(s: SimpleNamespace) -> Expression:
        """Return the Rayleigh dissipation of the viscous friction of the rotor."""
        dissipation: Expression = s.target.b * derivative(s.target.phi, 1) ** 2 / 2
        return dissipation

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        return CompositeSystem(
            self.label,
            elements=[
                VoltageSourceElement('source', V=self.V),
                ResistorElement('resistor', R=self.R),
                InductorElement('coil', L=self.L),
                ArmatureElement('winding'),
                GroundElement('ground'),
                InertiaElement('rotor', J=self.J, b=self.b),
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
                    lagrangian=self._get_rotor_lagrangian,
                    dissipation=self._get_rotor_dissipation,
                ),
                MotorCouplingInteraction('motor', armature='winding', inertia='rotor', k=self.k),
            ],
        )


def _get_spring_energy(s: SimpleNamespace) -> Expression:
    """Return the energy of the suspension spring and of the tire on the road step."""
    road = step(s.t, s.car.at, s.car.height)
    energy: Expression = s.car.k * (s.car.y1 - s.car.y2) ** 2 / 2 + s.car.kt * (s.car.y2 - road) ** 2 / 2
    return energy


class QuarterCarSystem(BaseSystem):
    """A quarter-car suspension driving over a road step.

    A body of mass `m1` rests on a spring and a damper above a wheel of mass `m2`, and a tire spring joins the wheel to
    the road, which steps to `height` at time `at`. Both displacements settle at the height of the step. The system has
    the displacements `car.y1` of the body and `car.y2` of the wheel.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.mechatronics import QuarterCarSystem
        >>> QuarterCarSystem(height=0.1).prepare().system_.elements[0].height
        0.1
    """

    field = 'engineering/mechatronics'
    topics = ('mechatronics', 'suspension', 'mass spring damper', 'ode', 'second order')
    references = ('Fritzson 2015',)
    until = 10.0
    dt = 0.005
    observables = (ObservableSpec('energy', _get_spring_energy),)

    def __init__(
        self,
        height: float = 0.05,
        *,
        at: float = 0.5,
        m1: float = 250.0,
        m2: float = 30.0,
        k: float = 16000.0,
        b: float = 1000.0,
        kt: float = 160000.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.height = height
        self.at = at
        self.m1 = m1
        self.m2 = m2
        self.k = k
        self.b = b
        self.kt = kt

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        car = QuarterCarElement(
            'car',
            m1=self.m1,
            m2=self.m2,
            k=self.k,
            b=self.b,
            kt=self.kt,
            height=self.height,
            at=self.at,
        )
        suspension = SpringDamperInteraction(
            'suspension',
            target='car',
            states=('y1', 'y2'),
            stiffness='k',
            damping='b',
            divisor={'y1': 'm1', 'y2': 'm2'},
        )
        return CompositeSystem(self.label, elements=[car], interactions=[suspension])
