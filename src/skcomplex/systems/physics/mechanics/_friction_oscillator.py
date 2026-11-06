"""Mass on a spring with static, Coulomb, and viscous friction, which sticks and slips."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import both, where
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class FrictionOscillatorElement(Element):
    """Mass on a spring with static, Coulomb, and viscous friction, which sticks and slips.

    The mass `m` at the position `x` with the velocity `v` feels the spring `-k x` and the viscous friction `-c v`.
    While it slides it feels the Coulomb friction `-mu_k m g sign(v)`, and at rest it stays at rest while the spring
    force stays within the static friction `mu_s m g`. The friction is discontinuous in the velocity, so the mass
    counts as at rest below the speed `v_stick`, which must exceed the change of speed of one solver step, and its
    position stays still while it is at rest.

    Args:
        label:
            The label of the mass.

        visual:
            How the mass is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        v:
            The initial velocity.

        m:
            The mass.

        k:
            The stiffness of the spring.

        c:
            The viscous friction coefficient.

        mu_s:
            The static friction coefficient.

        mu_k:
            The kinetic friction coefficient.

        g:
            The gravitational acceleration.

        v_stick:
            The speed below which the mass counts as at rest.

    Attributes:
        default_visual_:
            The visual of the mass when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import FrictionOscillatorElement
        >>> FrictionOscillatorElement('mass').mu_s
        0.06
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.03,
        v: Annotated[float, StateSpec()] = 0.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        c: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        mu_s: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.06,
        mu_k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        v_stick: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v
        self.m = m
        self.k = k
        self.c = c
        self.mu_s = mu_s
        self.mu_k = mu_k
        self.g = g
        self.v_stick = v_stick

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the motion of the mass under the spring and the friction that sticks and slips.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rates of the position and the velocity.
        """
        s = namespace
        force = -s.k * s.x - s.c * s.v
        moving = abs(s.v) > s.v_stick
        direction = where(moving, where(s.v > 0, 1, -1), where(force > 0, 1, -1))
        stuck = both(abs(s.v) <= s.v_stick, abs(s.k * s.x) <= s.mu_s * s.m * s.g)
        return [
            RateLaw(s.x, where(moving, s.v, 0)),
            RateLaw(s.v, where(stuck, 0, (force - s.mu_k * s.m * s.g * direction) / s.m)),
        ]
