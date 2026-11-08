"""Magnetic levitation: a conductive ball in a vacuum.

Alternating impulses of a field push and pull the ball.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import ImpulseInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.system import BaseSystem, System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class LevitatedBallElement(Element):
    """Conductive ball levitated in a vacuum.

    Between impulses the velocity follows `v' = -b v`, and every impulse of the field changes it by `m` or `-m`.

    Args:
        label:
            The label of the ball.

        visual:
            How the ball is drawn, or `None` for `default_visual_`.

        v:
            The initial velocity.

        b:
            The damping rate.

        m:
            The strength of every impulse.

    Examples:
        >>> from skcomplex.systems.engineering.mechatronics import LevitatedBallElement
        >>> LevitatedBallElement('ball').m
        0.2
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        v: Annotated[float, StateSpec()] = 0.0,
        b: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        m: Annotated[float, ParameterSpec()] = 0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.v = v
        self.b = b
        self.m = m

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the relaxation of the velocity.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.v, -s.b * s.v)]


IMPULSE_PERIOD = 4.0


class MagneticLevitationSystem(BaseSystem):
    """A conductive ball levitated in a vacuum by a field.

    The field pushes and pulls it with alternating impulses. The velocity follows
    `v' + b v = m Σ_k [δ(t + 1 - 4 k) - δ(t - 1 - 4 k)]`: it relaxes freely, rises by `m` at the times `4 k - 1`, and
    falls by `m` at the times `4 k + 1`. Its periodic solution, with the triangular wave
    `τ(t) = (2 / π) arcsin(sin(π t / 2))` and the rectangular wave `e(t) = sign(cos(π t / 2))`, is
    `v = m / (2 cosh b) exp(-b τ(t) e(t)) e(t)`, which the ball follows from the start when it starts at
    `m / (2 cosh b)`. The system has the velocity `ball.v`, raised by the impulses `push` and lowered by the impulses
    `pull`.

    Args:
        v:
            The initial velocity, or `None` for the value `m / (2 cosh b)` of the periodic solution at time 0.

        b:
            The damping rate.

        m:
            The strength of every impulse.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.mechatronics import MagneticLevitationSystem
        >>> round(MagneticLevitationSystem(b=1.0, m=0.2).prepare().system_.elements[0].v, 4)
        0.0648
    """

    field = 'engineering/mechatronics'
    topics = ('mechatronics', 'levitation', 'impulses', 'events', 'ode')
    references = ('Pilipchuk 2010', 'Trzaska 2023')
    until = 20.0
    dt = 0.01

    def __init__(
        self,
        v: float | None = None,
        *,
        b: float = 1.0,
        m: float = 0.2,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.v = v
        self.b = b
        self.m = m

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        start = self.m / (2 * math.cosh(self.b)) if self.v is None else self.v
        push = {'v': lambda s: s.target.v + s.target.m}
        pull = {'v': lambda s: s.target.v - s.target.m}
        return System(
            self.label,
            elements=[LevitatedBallElement('ball', v=start, b=self.b, m=self.m)],
            interactions=[
                ImpulseInteraction('push', target='ball', reset=push, period=IMPULSE_PERIOD, start=3.0),
                ImpulseInteraction('pull', target='ball', reset=pull, period=IMPULSE_PERIOD, start=1.0),
            ],
        )
