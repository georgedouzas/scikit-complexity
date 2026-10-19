"""Neuroscience models: adaptive synapses with memory on several time scales."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, state
from ....core.model import Element, Law, Symbols, System, Update, mod, where
from ....core.runtime import register_model


class Adaptrode(Element):
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
        >>> from skcomplex.models.biology.neuroscience import Adaptrode
        >>> Adaptrode('synapse').w_max
        1.0
    """

    w0: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    w1: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    w2: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    w_max: float = parameter(default=1.0, domain=Domain.POSITIVE)
    alpha0: float = parameter(default=0.25, domain=Domain.NONNEGATIVE)
    alpha1: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)
    alpha2: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)
    delta0: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    delta1: float = parameter(default=0.02, domain=Domain.NONNEGATIVE)
    delta2: float = parameter(default=0.002, domain=Domain.NONNEGATIVE)
    amplitude: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    period: float = parameter(default=10.0, domain=Domain.POSITIVE)
    width: float = parameter(default=3.0, domain=Domain.NONNEGATIVE)
    stop: float = parameter(default=150.0)

    def laws(self, s: Symbols) -> list[Law]:
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
            Update(s.w0, s.w0 + s.alpha0 * x * (s.w_max - s.w0) - s.delta0 * (s.w0 - s.w1)),
            Update(s.w1, s.w1 + s.alpha1 * (s.w0 - s.w1) - s.delta1 * (s.w1 - s.w2)),
            Update(s.w2, s.w2 + s.alpha2 * (s.w1 - s.w2) - s.delta2 * s.w2),
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
    synapse = Adaptrode(
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
