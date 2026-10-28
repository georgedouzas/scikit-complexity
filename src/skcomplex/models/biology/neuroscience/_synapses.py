"""Neuroscience models: adaptive synapses with memory on several time scales."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, UpdateLaw
from ....core.laws.utils import mod, where
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model


class AdaptrodeElement(Element):
    """Adaptrode of Mobus, a synaptic weight with three levels of memory that learn on increasing time scales.

    At every step the fast weight `w0` rises toward `w_max` in proportion to the input `x` and decays toward the
    slower weight `w1`, which rises toward `w0` and decays toward the slowest weight `w2`, which rises toward `w1` and
    decays toward zero. The input is a train of pulses of height `amplitude`, `width` steps long, every `period`
    steps, until the step `stop`, so the weights record recent, medium-term, and long-term activity.

    Args:
        label:
            The label of the synapse.

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
        >>> from skcomplex.models.biology.neuroscience import AdaptrodeElement
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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the updates of the three weights driven by the pulse train.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        x = where(mod(s.t, s.period) < s.width, where(s.t < s.stop, s.amplitude, 0), 0)
        return [
            UpdateLaw(s.w0, s.w0 + s.alpha0 * x * (s.w_max - s.w0) - s.delta0 * (s.w0 - s.w1)),
            UpdateLaw(s.w1, s.w1 + s.alpha1 * (s.w0 - s.w1) - s.delta1 * (s.w1 - s.w2)),
            UpdateLaw(s.w2, s.w2 + s.alpha2 * (s.w1 - s.w2) - s.delta2 * s.w2),
        ]


@register_model(
    field='biology/neuroscience',
    tags=('synapse', 'learning', 'memory', 'discrete map'),
    references=('Mobus 1994', 'Mobus 2022'),
    defaults={'until': 400, 'dt': 1},
)
def adaptrode(
    *,
    alpha0: float = 0.25,
    alpha1: float = 0.05,
    alpha2: float = 0.01,
    delta0: float = 0.1,
    delta1: float = 0.02,
    delta2: float = 0.002,
    period: float = 10.0,
    stop: float = 150.0,
) -> System:
    """Return an Adaptrode synapse driven by a train of input pulses that stops after a while.

    While the pulses arrive the fast weight follows each pulse, the medium weight builds up, and the slow weight
    builds up further. After the input stops the fast weight falls back quickly to the medium one, which keeps a
    trace of the activity far longer, while the slow weight retains a long-term memory.

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

    Returns:
        system:
            The model, with the weights `synapse.w0`, `synapse.w1`, and `synapse.w2`.

    Examples:
        >>> from skcomplex.models.biology.neuroscience import adaptrode
        >>> adaptrode(stop=50.0).elements[0].stop
        50.0
    """
    synapse = AdaptrodeElement(
        'synapse',
        alpha0=alpha0,
        alpha1=alpha1,
        alpha2=alpha2,
        delta0=delta0,
        delta1=delta1,
        delta2=delta2,
        period=period,
        stop=stop,
    )
    return System(elements=[synapse])
