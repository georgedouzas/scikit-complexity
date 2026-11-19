"""Coupled Colpitts oscillators: two detuned chaotic oscillators.

They lock their phases through a coupling.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import where
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class ColpittsOscillatorElement(Element):
    """Colpitts oscillator in dimensionless form.

    Its transistor conducts below a threshold of its third variable. The oscillator follows `x' = omega y - a F(z)`,
    `y' = c - omega x - z - b y`, and `z' = epsilon (y - d)`, with the transistor characteristic
    `F(z) = max(0, -1 - z)`, which is chaotic for `a = 30`, `b = 1`, `c = 20`, `d = 1`, and `epsilon = 1`.

    Args:
        label:
            The label of the oscillator.

        x:
            The initial first variable.

        y:
            The initial second variable.

        z:
            The initial third variable.

        omega:
            The natural frequency.

        a:
            The gain of the transistor.

        b:
            The damping of the second variable.

        c:
            The bias of the second variable.

        d:
            The offset of the third variable.

        epsilon:
            The time scale of the third variable.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import ColpittsOscillatorElement
        >>> ColpittsOscillatorElement('oscillator').a
        30.0
    """

    def __init__(
        self,
        label: str,
        *,
        x: float = 0.1,
        y: float = 0.0,
        z: float = 0.0,
        omega: Annotated[float, RealSet.open(0, oo)] = 1.0,
        a: Annotated[float, RealSet.closed_open(0, oo)] = 30.0,
        b: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        c: float = 20.0,
        d: float = 1.0,
        epsilon: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.z = z
        self.omega = omega
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.epsilon = epsilon


class ColpittsDynamicsInteraction(Interaction):
    """Dynamics of one Colpitts oscillator, with the constants the oscillator holds.

    It sets `x' = omega y - a F(z)`, `y' = c - omega x - z - b y`, and `z' = epsilon (y - d)`, with
    `F(z) = max(0, -1 - z)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the oscillator.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import ColpittsDynamicsInteraction
        >>> ColpittsDynamicsInteraction('dynamics', target='oscillator').target
        'oscillator'
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the three variables of the oscillator.

        Args:
            context:
                The space the interaction inherits from its system, or the namespace of the oscillator.
        """
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        o, t = context.target, context.t
        conduction = where(o.z < -1, -1 - o.z, 0)
        self.laws_: list[BaseComponent] = [
            RateLaw(o.x.diff(t), o.omega * o.y - o.a * conduction),
            RateLaw(o.y.diff(t), o.c - o.omega * o.x - o.z - o.b * o.y),
            RateLaw(o.z.diff(t), o.epsilon * (o.y - o.d)),
        ]


class ColpittsCouplingInteraction(Interaction):
    """Coupling of a Colpitts oscillator to another through their second variables.

    It adds `y_t' += k (y_s - y_t)`.

    Args:
        label:
            The label of the interaction.

        source:
            The path of the oscillator whose second variable the coupling reads.

        target:
            The path of the oscillator whose second variable the coupling changes.

        k:
            The coupling strength.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import ColpittsCouplingInteraction
        >>> ColpittsCouplingInteraction('coupling', source='a', target='b', k=0.4).k
        0.4
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        source: Role,
        target: Role,
        k: Annotated[float, RealSet.closed_open(0, oo)] = 0.4,
    ) -> None:
        super().__init__(label)
        self.source = source
        self.target = target
        self.k = k

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate `k (y_s - y_t)` of the second variable of the target.

        Args:
            context:
                The space the interaction inherits from its system, or the namespace of both oscillators.
        """
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        y = context.target.y
        self.laws_: list[BaseComponent] = [RateLaw(y.diff(context.t), context.k * (context.source.y - y))]


class CoupledColpittsSystem(BaseSystem):
    """Two detuned chaotic Colpitts oscillators.

    They are coupled through the difference of their second variables. Oscillator `i` follows `x' = omega_i y - a F(z)`,
    `y' = c - omega_i x - z - b y + k (y_j - y_i)`, and `z' = epsilon (y - d)`, with the transistor characteristic
    `F(z) = max(0, -1 - z)`. The phase of an oscillator is the angle of its orbit in the plane of `y` and `z`, around
    the means of both. Without coupling the two oscillators run at different mean frequencies and their phases drift
    apart, while for `k` near 0.3 to 0.5 their phases lock although their amplitudes stay irregular.

    The system has the states `oscillator_0.x`, `oscillator_0.y`, `oscillator_0.z`, and the same states of
    `oscillator_1`.

    Args:
        k:
            The coupling strength.

        omegas:
            The natural frequencies of the two oscillators.

        a:
            The gain of the transistor.

        b:
            The damping of the second variable.

        c:
            The bias of the second variable.

        d:
            The offset of the third variable.

        epsilon:
            The time scale of the third variable.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import CoupledColpittsSystem
        >>> [element.omega for element in CoupledColpittsSystem().prepare().system_.elements]
        [1.02, 0.98]
    """

    field = 'engineering/circuits'
    topics = ('circuit', 'synchronization', 'chaos', 'ode')
    references = ('Kennedy 1994', 'Datseris and Parlitz 2022')
    until = 500.0
    dt = 0.05

    def __init__(
        self,
        k: float = 0.4,
        *,
        omegas: tuple[float, float] = (1.02, 0.98),
        a: float = 30.0,
        b: float = 1.0,
        c: float = 20.0,
        d: float = 1.0,
        epsilon: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.k = k
        self.omegas = omegas
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.epsilon = epsilon

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        starts = ((0.1, 0.0, 0.0), (0.0, 0.1, 0.2))
        oscillators = [
            ColpittsOscillatorElement(
                f'oscillator_{index}',
                x=start[0],
                y=start[1],
                z=start[2],
                omega=omega,
                a=self.a,
                b=self.b,
                c=self.c,
                d=self.d,
                epsilon=self.epsilon,
            )
            for index, (omega, start) in enumerate(zip(self.omegas, starts, strict=True))
        ]
        dynamics = [
            ColpittsDynamicsInteraction(f'dynamics_{index}', target=f'oscillator_{index}') for index in range(2)
        ]
        couplings = [
            ColpittsCouplingInteraction(f'coupling_{target}', source=source, target=target, k=self.k)
            for source, target in (('oscillator_1', 'oscillator_0'), ('oscillator_0', 'oscillator_1'))
        ]
        self.system_ = System(self.label, elements=oscillators, interactions=[*dynamics, *couplings])
