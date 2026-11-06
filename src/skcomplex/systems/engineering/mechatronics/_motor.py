"""Armature, inertia, and the coupling of a DC motor between a circuit and a Lagrangian coordinate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, EquationLaw
from ....core.components.laws.utils import derivative
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..common.electrical import TwoPinElement

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
