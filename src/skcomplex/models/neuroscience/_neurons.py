"""Model spiking neurons whose spikes are events that reset the membrane potential."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...core import Domain, parameter
from ...model import Element, Event, Rate, System
from ...runtime import register_model

LeakyIntegrateAndFire = Element.define(
    'LeakyIntegrateAndFire',
    states={'v': -65.0},
    parameters={
        'rest': -65.0,
        'threshold': -50.0,
        'reset': -65.0,
        'resistance': parameter(default=10.0, domain=Domain.POSITIVE),
        'tau': parameter(default=10.0, domain=Domain.POSITIVE),
        'I': 2.0,
    },
    laws=lambda s: [
        Rate(s.v, (-(s.v - s.rest) + s.resistance * s.I) / s.tau),
        Event(s.v - s.threshold, {s.v: s.reset}, direction=1, label='spike'),
    ],
)

Izhikevich = Element.define(
    'Izhikevich',
    states={'v': -65.0, 'u': -13.0},
    parameters={'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0, 'I': 10.0, 'peak': 30.0},
    laws=lambda s: [
        Rate(s.v, 0.04 * s.v**2 + 5 * s.v + 140 - s.u + s.I),
        Rate(s.u, s.a * (s.b * s.v - s.u)),
        Event(s.v - s.peak, {s.v: s.c, s.u: s.u + s.d}, direction=1, label='spike'),
    ],
)


@register_model(
    field='neuroscience',
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
        >>> from skcomplex.models.neuroscience import integrate_and_fire
        >>> from skcomplex.runtime import Simulation
        >>> trajectory = Simulation(integrate_and_fire(), dt=0.1).run(until=50.0).trajectory_
        >>> round(trajectory.events[0].time, 4)
        13.8629
    """
    return System(
        elements=[
            LeakyIntegrateAndFire(
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
    field='neuroscience',
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
        >>> from skcomplex.models.neuroscience import izhikevich
        >>> izhikevich(d=2.0).elements[0].d
        2.0
    """
    return System(elements=[Izhikevich('neuron', v=v, u=b * v if u is None else u, a=a, b=b, c=c, d=d, I=I)])
