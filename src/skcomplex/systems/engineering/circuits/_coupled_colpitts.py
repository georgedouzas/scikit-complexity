"""Coupled Colpitts oscillators: two detuned chaotic oscillators.

They lock their phases through a coupling.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import RateInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import where
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec


class ColpittsOscillatorElement(Element):
    """Colpitts oscillator in dimensionless form.

    Its transistor conducts below a threshold of its third variable. The oscillator follows `x' = omega y - a F(z)`,
    `y' = c - omega x - z - b y`, and `z' = epsilon (y - d)`, with the transistor characteristic
    `F(z) = max(0, -1 - z)`, which is chaotic for `a = 30`, `b = 1`, `c = 20`, `d = 1`, and `epsilon = 1`.

    Args:
        label:
            The label of the oscillator.

        visual:
            How the oscillator is drawn, or `None` for `default_visual_`.

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

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.1,
        y: Annotated[float, StateSpec()] = 0.0,
        z: Annotated[float, StateSpec()] = 0.0,
        omega: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 30.0,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        c: Annotated[float, ParameterSpec()] = 20.0,
        d: Annotated[float, ParameterSpec()] = 1.0,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.omega = omega
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.epsilon = epsilon

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the three variables of the oscillator.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        conduction = where(s.z < -1, -1 - s.z, 0)
        return [
            RateLaw(s.x, s.omega * s.y - s.a * conduction),
            RateLaw(s.y, s.c - s.omega * s.x - s.z - s.b * s.y),
            RateLaw(s.z, s.epsilon * (s.y - s.d)),
        ]


class ColpittsCouplingInteraction(RateInteraction):
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

    def __init__(
        self,
        label: str,
        *,
        source: Annotated[str, RoleSpec()],
        target: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.4,
    ) -> None:
        super().__init__(label, target=target, state='y')
        self.source = source
        self.k = k

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return `k (y_s - y_t)`.

        Args:
            namespace:
                The namespace of the symbols the rate can use.

        Returns:
            rate:
                The rate of the second variable of the target.
        """
        rate: Expression = namespace.k * (namespace.source.y - namespace.target.y)
        return rate


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
    ) -> None:
        super().__init__(label)
        self.k = k
        self.omegas = omegas
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.epsilon = epsilon

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
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
        couplings = [
            ColpittsCouplingInteraction(f'coupling_{target}', source=source, target=target, k=self.k)
            for source, target in (('oscillator_1', 'oscillator_0'), ('oscillator_0', 'oscillator_1'))
        ]
        return CompositeSystem(self.label, elements=oscillators, interactions=couplings)
