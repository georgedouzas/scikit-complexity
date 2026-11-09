"""Neuroscience systems: adaptive synapses with memory on several time scales."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import MapInteraction
from ....core.components.laws.utils import mod, where
from ....core.components.systems import BaseSystem, System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class AdaptrodeElement(Element):
    """Adaptrode of Mobus, a synaptic weight with three levels of memory.

    The levels learn on increasing time scales.

    At every step the fast weight `w0` rises toward `w_max` in proportion to the input `x` and decays toward the slower
    weight `w1`, which rises toward `w0` and decays toward the slowest weight `w2`, which rises toward `w1` and decays
    toward zero. The input is a train of pulses of height `amplitude`, `width` steps long, every `period` steps, until
    the step `stop`, so the weights record recent, medium-term, and long-term activity. The element holds the states and
    parameters and has no laws of its own: the maps of `AdaptrodeSystem`, one per level, read its parameters.

    Args:
        label:
            The label of the synapse.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        w0:
            The initial fast weight.

        w1:
            The initial medium weight.

        w2:
            The initial slow weight.

        w_max:
            The largest weight.

        alpha0:
            The learning rate of the fast weight.

        alpha1:
            The learning rate of the medium weight.

        alpha2:
            The learning rate of the slow weight.

        delta0:
            The decay rate of the fast weight.

        delta1:
            The decay rate of the medium weight.

        delta2:
            The decay rate of the slow weight.

        amplitude:
            The height of the input pulses.

        period:
            The number of steps between the starts of two pulses.

        width:
            The number of steps of each pulse.

        stop:
            The step after which the input stays off.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import AdaptrodeElement
        >>> AdaptrodeElement('synapse').w_max
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        w0: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        w1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        w2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        w_max: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        alpha0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.25,
        alpha1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        alpha2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        delta0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        delta1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
        delta2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.002,
        amplitude: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        period: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        width: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
        stop: Annotated[float, ParameterSpec()] = 150.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.w_max = w_max
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.delta0 = delta0
        self.delta1 = delta1
        self.delta2 = delta2
        self.amplitude = amplitude
        self.period = period
        self.width = width
        self.stop = stop


class AdaptrodeSystem(BaseSystem):
    """An Adaptrode synapse driven by a train of input pulses that stops after a while.

    While the pulses arrive the fast weight follows each pulse, the medium weight builds up, and the slow weight builds
    up further. After the input stops the fast weight falls back quickly to the medium one, which keeps a trace of the
    activity far longer, while the slow weight retains a long-term memory. The system has the weights `synapse.w0`,
    `synapse.w1`, and `synapse.w2`.

    Args:
        alpha0:
            The learning rate of the fast weight.

        alpha1:
            The learning rate of the medium weight.

        alpha2:
            The learning rate of the slow weight.

        delta0:
            The decay rate of the fast weight.

        delta1:
            The decay rate of the medium weight.

        delta2:
            The decay rate of the slow weight.

        period:
            The number of steps between the starts of two pulses.

        stop:
            The step after which the input stays off.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import AdaptrodeSystem
        >>> AdaptrodeSystem(stop=50.0).prepare().system_.elements[0].stop
        50.0
    """

    field = 'biology/neuroscience'
    topics = ('synapse', 'learning', 'memory', 'discrete map')
    references = ('Mobus 1994', 'Mobus 2022')
    until = 400
    dt = 1

    def __init__(
        self,
        *,
        alpha0: float = 0.25,
        alpha1: float = 0.05,
        alpha2: float = 0.01,
        delta0: float = 0.1,
        delta1: float = 0.02,
        delta2: float = 0.002,
        period: float = 10.0,
        stop: float = 150.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.delta0 = delta0
        self.delta1 = delta1
        self.delta2 = delta2
        self.period = period
        self.stop = stop

    @staticmethod
    def _input(s: SimpleNamespace) -> Expression:
        """Return the input pulse train of the Adaptrode.

        The train is `amplitude` for `width` steps every `period` until `stop`.
        """
        synapse = s.target
        pulse: Expression = where(
            mod(s.t, synapse.period) < synapse.width,
            where(s.t < synapse.stop, synapse.amplitude, 0),
            0,
        )
        return pulse

    def _build_system(self) -> System:
        """Return the parts of an Adaptrode synapse driven by a train of input pulses.

        The train stops after a while.
        """
        synapse = AdaptrodeElement(
            'synapse',
            alpha0=self.alpha0,
            alpha1=self.alpha1,
            alpha2=self.alpha2,
            delta0=self.delta0,
            delta1=self.delta1,
            delta2=self.delta2,
            period=self.period,
            stop=self.stop,
        )
        levels = [
            MapInteraction(
                'fast_level',
                target='synapse',
                state='w0',
                function=lambda s: s.target.w0
                + s.target.alpha0 * self._input(s) * (s.target.w_max - s.target.w0)
                - s.target.delta0 * (s.target.w0 - s.target.w1),
            ),
            MapInteraction(
                'medium_level',
                target='synapse',
                state='w1',
                function=lambda s: s.target.w1
                + s.target.alpha1 * (s.target.w0 - s.target.w1)
                - s.target.delta1 * (s.target.w1 - s.target.w2),
            ),
            MapInteraction(
                'slow_level',
                target='synapse',
                state='w2',
                function=lambda s: s.target.w2
                + s.target.alpha2 * (s.target.w1 - s.target.w2)
                - s.target.delta2 * s.target.w2,
            ),
        ]
        return System(self.label, elements=[synapse], interactions=levels)
