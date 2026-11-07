"""Neuroscience systems: spiking neurons."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.system import System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system
from ..common.excitable import SpikeResetInteraction

PEAK = 30.0


class IzhikevichElement(Element):
    """Izhikevich neuron, a quadratic potential `v` and a linear recovery `u`.

    The potential follows `v' = 0.04 v² + 5 v + 140 - u + I` and the recovery `u' = a (b v - u)`. A spike reset fires
    the neuron.

    Args:
        label:
            The label of the neuron.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        v:
            The initial potential.

        u:
            The initial recovery.

        a:
            The time scale of the recovery.

        b:
            The sensitivity of the recovery to the potential.

        I:
            The input current.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import IzhikevichElement
        >>> IzhikevichElement('neuron').b
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        v: Annotated[float, StateSpec()] = -65.0,
        u: Annotated[float, StateSpec()] = -13.0,
        a: Annotated[float, ParameterSpec()] = 0.02,
        b: Annotated[float, ParameterSpec()] = 0.2,
        I: Annotated[float, ParameterSpec()] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.v = v
        self.u = u
        self.a = a
        self.b = b
        self.I = I

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the quadratic potential and the linear recovery.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.v, 0.04 * s.v**2 + 5 * s.v + 140 - s.u + s.I), RateLaw(s.u, s.a * (s.b * s.v - s.u))]


class LeakyIntegrateAndFireElement(Element):
    """Leaky integrate-and-fire neuron, whose potential relaxes to `rest + resistance I` over the time `tau`.

    The potential follows `tau v' = -(v - rest) + resistance I`. A spike reset fires the neuron. Units are mV, ms, MΩ,
    and nA.

    Args:
        label:
            The label of the neuron.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        v:
            The initial potential.

        rest:
            The resting potential.

        resistance:
            The membrane resistance.

        tau:
            The membrane time constant.

        I:
            The input current.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import LeakyIntegrateAndFireElement
        >>> LeakyIntegrateAndFireElement('neuron').tau
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        v: Annotated[float, StateSpec()] = -65.0,
        rest: Annotated[float, ParameterSpec()] = -65.0,
        resistance: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        I: Annotated[float, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.v = v
        self.rest = rest
        self.resistance = resistance
        self.tau = tau
        self.I = I

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the leaky integration of the input current.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.v, (-(s.v - s.rest) + s.resistance * s.I) / s.tau)]


@register_system(
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
            The system, with the membrane potential `neuron.v` and its spikes recorded as `spike` events.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import integrate_and_fire
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> trajectory = Simulation(integrate_and_fire(), dt=0.1).prepare().run(until=50.0).trajectory_
        >>> round(trajectory.events[0].time, 2)
        13.86
    """
    neuron = LeakyIntegrateAndFireElement('neuron', v=v, rest=rest, resistance=resistance, tau=tau, I=I)
    spike = SpikeResetInteraction('spike', target='neuron', threshold=threshold, after=reset)
    return System(elements=[neuron], interactions=[spike])


@register_system(
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
            The system, with the potential `neuron.v`, the recovery `neuron.u`, and its spikes as `spike` events.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import izhikevich
        >>> izhikevich(d=2.0).interactions[0].kick
        2.0
    """
    neuron = IzhikevichElement('neuron', v=v, u=b * v if u is None else u, a=a, b=b, I=I)
    spike = SpikeResetInteraction('spike', target='neuron', threshold=PEAK, after=c, kicked='u', kick=d)
    return System(elements=[neuron], interactions=[spike])
