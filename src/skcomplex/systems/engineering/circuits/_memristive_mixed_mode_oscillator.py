"""Memristive mixed-mode oscillator.

It is a slow-fast circuit with a cubic flux-controlled memristor.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import ScipyODESolver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class MemristiveCircuitElement(Element):
    """Slow-fast circuit with a cubic flux-controlled memristor.

    Its oscillations mix small and large amplitudes. The memristor has the charge `q = a w + b w^3` of its flux `w`, so
    its memductance is `a + 3 b w^2`, and the circuit follows `epsilon x' = -y - (a + 3 b w^2) x`,
    `y' = alpha (x - R y - z + e)`, `z' = -β y`, and `w' = x`.

    Args:
        label:
            The label of the circuit.

        visual:
            How the circuit is drawn, or `None` for `default_visual_`.

        x:
            The initial voltage of the memristor.

        y:
            The initial current of the inductor.

        z:
            The initial voltage of the second capacitor.

        w:
            The initial flux of the memristor.

        epsilon:
            The first capacitance, which sets the fast time scale.

        R:
            The resistance.

        alpha:
            The inverse inductance.

        beta:
            The gain of the controlled source over the second capacitance.

        e:
            The voltage of the source.

        a:
            The linear coefficient of the memristor characteristic.

        b:
            The cubic coefficient of the memristor characteristic.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import MemristiveCircuitElement
        >>> MemristiveCircuitElement('circuit').a
        -0.34
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 0.0,
        z: Annotated[float, StateSpec()] = 0.0,
        w: Annotated[float, StateSpec()] = 0.0,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
        R: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.55,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.15,
        e: Annotated[float, ParameterSpec()] = -0.04,
        a: Annotated[float, ParameterSpec()] = -0.34,
        b: Annotated[float, ParameterSpec()] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.epsilon = epsilon
        self.R = R
        self.alpha = alpha
        self.beta = beta
        self.e = e
        self.a = a
        self.b = b

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the voltages, the current, and the flux of the circuit.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            RateLaw(s.x, -(s.y + (s.a + 3 * s.b * s.w**2) * s.x) / s.epsilon),
            RateLaw(s.y, s.alpha * (s.x - s.R * s.y - s.z + s.e)),
            RateLaw(s.z, -s.beta * s.y),
            RateLaw(s.w, s.x),
        ]


class MemristiveMixedModeOscillatorSystem(BaseSystem):
    """A slow-fast circuit with a cubic flux-controlled memristor.

    Its oscillations mix two amplitudes. The memristor has the charge `q = a w + b w^3` of its flux `w`, and with the
    voltage `x` of the memristor, the current `y` of the inductor, and the voltage `z` of the second capacitor, the
    circuit follows `epsilon x' = -y - (a + 3 b w^2) x`, `y' = alpha (x - R y - z + e)`, `z' = -β y`, and `w' = x`.
    Trzaska prints the memductance as `a + 3 b w`, but only the derivative `a + 3 b w^2` of the characteristic gives the
    response he reports: for the defaults, from rest, the voltage settles on the mixed-mode oscillation `2^11`, two
    large spikes and eleven small oscillations in a period of about 7.9, against the 7.816 he reports. The spikes last
    about a hundredth of the period, so the tuned solver runs at the relative tolerance `1e-8`, below which looser runs
    drift to an irregular pattern of shorter periods.

    The system has the states `circuit.x`, `circuit.y`, `circuit.z`, and `circuit.w`.

    Args:
        x:
            The initial voltage of the memristor.

        y:
            The initial current of the inductor.

        z:
            The initial voltage of the second capacitor.

        w:
            The initial flux of the memristor.

        epsilon:
            The first capacitance, which sets the fast time scale.

        R:
            The resistance.

        alpha:
            The inverse inductance.

        beta:
            The gain of the controlled source over the second capacitance.

        e:
            The voltage of the source.

        a:
            The linear coefficient of the memristor characteristic.

        b:
            The cubic coefficient of the memristor characteristic.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import MemristiveMixedModeOscillatorSystem
        >>> MemristiveMixedModeOscillatorSystem(a=-0.5).prepare().system_.elements[0].a
        -0.5
    """

    field = 'engineering/circuits'
    topics = ('circuit', 'memristor', 'mixed-mode oscillations', 'slow-fast', 'ode')
    references = ('Trzaska 2023',)
    until = 150.0
    dt = 0.02
    solver = ScipyODESolver(method='LSODA', rtol=1e-8, atol=1e-10)

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        w: float = 0.0,
        *,
        epsilon: float = 0.01,
        R: float = 2.0,
        alpha: float = 1.55,
        beta: float = 0.15,
        e: float = -0.04,
        a: float = -0.34,
        b: float = 0.05,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.epsilon = epsilon
        self.R = R
        self.alpha = alpha
        self.beta = beta
        self.e = e
        self.a = a
        self.b = b

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        circuit = MemristiveCircuitElement(
            'circuit',
            x=self.x,
            y=self.y,
            z=self.z,
            w=self.w,
            epsilon=self.epsilon,
            R=self.R,
            alpha=self.alpha,
            beta=self.beta,
            e=self.e,
            a=self.a,
            b=self.b,
        )
        return CompositeSystem(self.label, elements=[circuit])
