"""Magnetic levitation: a conductive ball in a vacuum.

Alternating impulses of a field push and pull the ball.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.signals import PulseInteraction

IMPULSE_PERIOD = 4.0


class LevitatedBallElement(Element):
    """Conductive ball levitated in a vacuum, with the velocity `v`.

    The damping and the impulses of `MagneticLevitationSystem` change the velocity.

    Args:
        label:
            The label of the ball.

        v:
            The initial velocity.

    Examples:
        >>> from skcomplex.systems.engineering.mechatronics import LevitatedBallElement
        >>> LevitatedBallElement('ball', v=0.1).v
        0.1
    """

    def __init__(self, label: str, *, v: float = 0.0) -> None:
        super().__init__(label)
        self.v = v


class DampingInteraction(Interaction):
    """Damping that relaxes the velocity of the target at the rate `b`, as `v' = -b v`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element whose velocity relaxes.

        b:
            The damping rate.

    Examples:
        >>> from skcomplex.systems.engineering.mechatronics import DampingInteraction
        >>> DampingInteraction('damping', target='ball').b
        1.0
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        b: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.b = b

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the relaxation of the velocity.

        Args:
            context:
                The space the interaction inherits from its system, or the namespace of the target.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            v = context.target.v
            self.laws_: list[BaseLaw] = [RateLaw(v.diff(context.t), -context.b * v)]


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.v = v
        self.b = b
        self.m = m

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_`, the ball, its damping, and the impulses that push and pull it."""
        super()._prepare(context)
        start = self.m / (2 * math.cosh(self.b)) if self.v is None else self.v
        push = {'v': lambda s: s.target.v + self.m}
        pull = {'v': lambda s: s.target.v - self.m}
        self.system_ = System(
            self.label,
            elements=[LevitatedBallElement('ball', v=start)],
            interactions=[
                DampingInteraction('damping', target='ball', b=self.b),
                PulseInteraction('push', target='ball', reset=push, period=IMPULSE_PERIOD, start=3.0),
                PulseInteraction('pull', target='ball', reset=pull, period=IMPULSE_PERIOD, start=1.0),
            ],
        )
