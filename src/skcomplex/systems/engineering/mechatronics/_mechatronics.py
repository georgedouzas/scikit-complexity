"""Mechatronic systems: a DC motor and a quarter-car suspension.

The DC motor is built from components and a Lagrangian.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import ConstraintLaw, RateLaw
from ....core.components.laws.utils import step
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ...general.mechanics import EulerLagrangeInteraction
from ..common.electrical import (
    BranchInteraction,
    GroundInteraction,
    InductorInteraction,
    ResistorInteraction,
    VoltageSourceInteraction,
)
from ..common.mechanics import SpringDamperInteraction

POSITIVE = RealSet.open(0, oo)


class InertiaElement(Element):
    """Rotating inertia of the angle `phi`, a coordinate of a Lagrangian.

    Args:
        label:
            The label of the element, unique among the components of its system.

        phi:
            The initial angle, whose derivative `phi'` is the angular velocity.

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
        *,
        phi: object = 0.0,
        J: Annotated[object, POSITIVE] = 1.0,
        b: Annotated[object, POSITIVE] = 1.0,
    ) -> None:
        super().__init__(label)
        self.phi = phi
        self.J = J
        self.b = b


class TireInteraction(Interaction):
    """Tire spring between a wheel and a road that steps up to `height` at the time `at`.

    The tire pushes the wheel with the force `kt (road - y)`, which accelerates the wheel divided by its mass.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the element that holds the wheel.

        state:
            The name of the displacement of the wheel.

        mass:
            The name of the mass of the wheel.

        kt:
            The stiffness of the tire.

        height:
            The height of the road step.

        at:
            The time of the road step.

    Examples:
        >>> from skcomplex.systems.engineering.mechatronics import TireInteraction
        >>> TireInteraction('tire', target='car', height=0.1).height
        0.1
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str = 'y2',
        mass: str = 'm2',
        kt: Annotated[float, POSITIVE] = 160000.0,
        height: float = 0.05,
        at: float = 0.5,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.mass = mass
        self.kt = kt
        self.height = height
        self.at = at

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of the wheel by the tire."""
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        wheel, t = getattr(context.target, self.state), context.t
        force = context.kt * (step(t, context.at, context.height) - wheel)
        self.laws_: list[BaseComponent] = [RateLaw(wheel.diff(t, 2), force / getattr(context.target, self.mass))]


class MotorCouplingInteraction(BranchInteraction):
    """Winding of a motor between the nodes `a` and `b` that drives an inertia with the motor constant `k`.

    The voltage of the winding is the back electromotive force `k phi'` of the angle `phi` of the rotor, and its current
    `i`, from `a` to `b`, exerts the torque `k i` on the angle, which adds to the Euler-Lagrange equation of the angle.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        a:
            The path of the node the current leaves.

        b:
            The path of the node the current enters.

        winding:
            The path of the element that holds the current `i` of the winding.

        rotor:
            The path of the inertia, whose angle is `phi`.

        k:
            The motor constant.

    Examples:
        >>> from skcomplex.systems.engineering.mechatronics import MotorCouplingInteraction
        >>> MotorCouplingInteraction('motor', a='terminal', b='neutral', winding='winding', rotor='rotor', k=0.5).k
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        a: Role,
        b: Role,
        winding: Role,
        rotor: Role,
        k: Annotated[float, POSITIVE] = 1.0,
    ) -> None:
        super().__init__(label, a=a, b=b)
        self.winding = winding
        self.rotor = rotor
        self.k = k

    def _get_branch_laws(self, s: SimpleNamespace) -> list[BaseComponent]:
        """Return the back electromotive force, the current into both balances, and the torque on the angle."""
        i, phi = s.winding.i, s.rotor.phi
        return [
            ConstraintLaw(s.a.v - s.b.v, s.k * phi.diff(s.t), target=i),
            ConstraintLaw(-i, 0, target=s.a.v),
            ConstraintLaw(i, 0, target=s.b.v),
            ConstraintLaw(-s.k * i, 0, target=phi),
        ]


def _get_speed_of_rotor(s: SimpleNamespace) -> Expression:
    """Return the angular velocity of the rotor."""
    speed: Expression = s.rotor.phi.diff(s.t)
    return speed


def _get_current_of_winding(s: SimpleNamespace) -> Expression:
    """Return the current of the winding."""
    current: Expression = s.winding.i
    return current


class DcMotorSystem(BaseSystem):
    """A DC motor whose armature circuit drives an inertia.

    The armature circuit is made of components and the inertia is on a Lagrangian coordinate. A voltage source drives a
    resistor, an inductor, and the armature in series. The armature has the voltage `k phi'`, and its current exerts the
    torque `k i` on the angle of an inertia with viscous friction, so the angular velocity settles at
    `k V / (R b + k^2)`. The system has the angle `rotor.phi`, the angular velocity `rotor.phi'`, and the current
    `winding.i`.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        Observable('speed', _get_speed_of_rotor),
        Observable('current', _get_current_of_winding),
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.V = V
        self.R = R
        self.L = L
        self.k = k
        self.J = J
        self.b = b

    @staticmethod
    def _get_rotor_lagrangian(s: SimpleNamespace) -> Expression:
        """Return the kinetic energy of the rotor."""
        energy: Expression = s.target.J * s.target.phi.diff(s.t) ** 2 / 2
        return energy

    @staticmethod
    def _get_rotor_dissipation(s: SimpleNamespace) -> Expression:
        """Return the Rayleigh dissipation of the viscous friction of the rotor."""
        dissipation: Expression = s.target.b * s.target.phi.diff(s.t) ** 2 / 2
        return dissipation

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[
                *[Element(name, values={'v': 0.0}) for name in ('supply', 'middle', 'terminal', 'neutral')],
                *[Element(name, values={'i': 0.0}) for name in ('source', 'coil', 'winding', 'earth')],
                InertiaElement('rotor', J=self.J, b=self.b),
            ],
            interactions=[
                VoltageSourceInteraction('mains', a='supply', b='neutral', branch='source', V=self.V),
                ResistorInteraction('resistor', a='supply', b='middle', R=self.R),
                InductorInteraction('inductor', a='middle', b='terminal', branch='coil', L=self.L),
                GroundInteraction('ground', node='neutral', branch='earth'),
                EulerLagrangeInteraction(
                    'rotation',
                    target='rotor',
                    states='phi',
                    lagrangian=self._get_rotor_lagrangian,
                    dissipation=self._get_rotor_dissipation,
                ),
                MotorCouplingInteraction(
                    'motor',
                    a='terminal',
                    b='neutral',
                    winding='winding',
                    rotor='rotor',
                    k=self.k,
                ),
            ],
        )


def _get_spring_energy(s: SimpleNamespace) -> Expression:
    """Return the energy of the suspension spring and of the tire on the road step."""
    road = step(s.t, s.tire.at, s.tire.height)
    energy: Expression = s.suspension.k * (s.car.y1 - s.car.y2) ** 2 / 2 + s.tire.kt * (s.car.y2 - road) ** 2 / 2
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.engineering.mechatronics import QuarterCarSystem
        >>> QuarterCarSystem(height=0.1).prepare().system_.interactions[0].height
        0.1
    """

    field = 'engineering/mechatronics'
    topics = ('mechatronics', 'suspension', 'mass spring damper', 'ode', 'second order')
    references = ('Fritzson 2015',)
    until = 10.0
    dt = 0.005
    observables = (Observable('energy', _get_spring_energy),)

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.height = height
        self.at = at
        self.m1 = m1
        self.m2 = m2
        self.k = k
        self.b = b
        self.kt = kt

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        car = Element('car', values={'y1': 0.0, 'y2': 0.0, 'm1': self.m1, 'm2': self.m2})
        suspension = SpringDamperInteraction(
            'suspension',
            target='car',
            states=('y1', 'y2'),
            masses=('m1', 'm2'),
            k=self.k,
            b=self.b,
        )
        tire = TireInteraction('tire', target='car', kt=self.kt, height=self.height, at=self.at)
        self.system_ = System(self.label, elements=[car], interactions=[tire, suspension])
