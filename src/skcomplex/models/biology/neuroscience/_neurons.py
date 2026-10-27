"""Neuroscience models: spiking neurons."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, CrossingLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.runtime import register_model


class LeakyIntegrateAndFireElement(Element):
    """Leaky integrate and fire element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        v: Annotated[object, StateSpec()] = -65.0,
        rest: Annotated[object, ParameterSpec()] = -65.0,
        threshold: Annotated[object, ParameterSpec()] = -50.0,
        reset: Annotated[object, ParameterSpec()] = -65.0,
        resistance: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        tau: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        I: Annotated[object, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.v = v
        self.rest = rest
        self.threshold = threshold
        self.reset = reset
        self.resistance = resistance
        self.tau = tau
        self.I = I

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the leaky integrate and fire element."""
        return [
            RateLaw(s.v, (-(s.v - s.rest) + s.resistance * s.I) / s.tau),
            CrossingLaw(s.v - s.threshold, {s.v: s.reset}, direction=1, label='spike'),
        ]


class IzhikevichElement(Element):
    """Izhikevich element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        v: Annotated[object, StateSpec()] = -65.0,
        u: Annotated[object, StateSpec()] = -13.0,
        a: Annotated[object, ParameterSpec()] = 0.02,
        b: Annotated[object, ParameterSpec()] = 0.2,
        c: Annotated[object, ParameterSpec()] = -65.0,
        d: Annotated[object, ParameterSpec()] = 8.0,
        I: Annotated[object, ParameterSpec()] = 10.0,
        peak: Annotated[object, ParameterSpec()] = 30.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.v = v
        self.u = u
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.I = I
        self.peak = peak

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the izhikevich element."""
        return [
            RateLaw(s.v, 0.04 * s.v**2 + 5 * s.v + 140 - s.u + s.I),
            RateLaw(s.u, s.a * (s.b * s.v - s.u)),
            CrossingLaw(s.v - s.peak, {s.v: s.c, s.u: s.u + s.d}, direction=1, label='spike'),
        ]


@register_model(
    field='biology/neuroscience',
    tags=('neuron', 'spiking', 'events'),
    references=('Lapicque 1907', 'Gerstner et al. 2014'),
    defaults={'until': 100.0, 'dt': 0.1},
)
def integrate_and_fire(
    v: float = -65.0,
    *,
    rest: float = -65.0,
    threshold: float = -50.0,
    reset: float = -65.0,
    resistance: float = 10.0,
    tau: float = 10.0,
    I: float = 2.0,
) -> System:
    """Return the leaky integrate-and-fire neuron, `tau v' = -(v - rest) + R I`, which spikes and resets at a threshold.

    With `R I` above `threshold - rest` it fires periodically with period `tau ln(R I / (R I - threshold + rest))`
    when `reset = rest`.

    Args:
        v:
            The initial membrane potential in mV.

        rest:
            The resting potential in mV.

        threshold:
            The potential at which the neuron spikes in mV.

        reset:
            The potential after a spike in mV.

        resistance:
            The membrane resistance in MΩ.

        tau:
            The membrane time constant in ms.

        I:
            The input current in nA.

    Returns:
        system:
            The model, with the membrane potential `neuron.v` and its spikes recorded as `spike` events.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import integrate_and_fire
        >>> from skcomplex.engine.runtime import Simulation
        >>> trajectory = Simulation(integrate_and_fire(), dt=0.1).run(until=50.0).trajectory_
        >>> round(trajectory.events[0].time, 2)
        13.86
    """
    return System(
        elements=[
            LeakyIntegrateAndFireElement(
                'neuron',
                v=v,
                rest=rest,
                threshold=threshold,
                reset=reset,
                resistance=resistance,
                tau=tau,
                I=I,
            ),
        ],
    )


@register_model(
    field='biology/neuroscience',
    tags=('neuron', 'spiking', 'events', 'bursting'),
    references=('Izhikevich 2003',),
    defaults={'until': 200.0, 'dt': 0.1},
)
def izhikevich(
    v: float = -65.0,
    u: float | None = None,
    *,
    a: float = 0.02,
    b: float = 0.2,
    c: float = -65.0,
    d: float = 8.0,
    I: float = 10.0,
) -> System:
    """Return the Izhikevich neuron, whose four parameters give regular spiking, bursting, chattering, and more.

    The potential follows `v' = 0.04 v² + 5 v + 140 - u + I` and the recovery `u' = a (b v - u)`. When `v` reaches
    30 mV the neuron spikes, `v` resets to `c`, and `u` grows by `d`. The defaults give regular spiking.

    Args:
        v:
            The initial membrane potential in mV.

        u:
            The initial recovery variable, or `None` for `b v`.

        a:
            The time scale of the recovery.

        b:
            The sensitivity of the recovery to the potential.

        c:
            The potential after a spike.

        d:
            The increase of the recovery after a spike.

        I:
            The input current.

    Returns:
        system:
            The model, with the potential `neuron.v`, the recovery `neuron.u`, and its spikes as `spike` events.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import izhikevich
        >>> izhikevich(d=2.0).elements[0].d
        2.0
    """
    return System(elements=[IzhikevichElement('neuron', v=v, u=b * v if u is None else u, a=a, b=b, c=c, d=d, I=I)])
