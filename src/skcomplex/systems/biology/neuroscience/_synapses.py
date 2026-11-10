"""Neuroscience systems: adaptive synapses with memory on several time scales."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.laws.utils import mod, where
from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


class AdaptrodeSystem(DynamicalSystem):
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
        >>> AdaptrodeSystem(stop=50.0).prepare().system_.elements[0].get_value('stop')
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
        super().__init__(label=label)
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.delta0 = delta0
        self.delta1 = delta1
        self.delta2 = delta2
        self.period = period
        self.stop = stop

    @staticmethod
    def _get_input(s: SimpleNamespace) -> Expression:
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

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'synapse'

    def _get_time(self) -> str:
        """Return the discrete time of the map."""
        return 'discrete'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the three weights and the parameters with their specs."""
        nonnegative = RealSet.closed_open(0, oo)
        return {
            'w0': DeclaredVariableSpec(0.0, StateSpec(domain=nonnegative)),
            'w1': DeclaredVariableSpec(0.0, StateSpec(domain=nonnegative)),
            'w2': DeclaredVariableSpec(0.0, StateSpec(domain=nonnegative)),
            'w_max': DeclaredVariableSpec(1.0, ParameterSpec(domain=RealSet.open(0, oo))),
            'alpha0': DeclaredVariableSpec(self.alpha0, ParameterSpec(domain=nonnegative)),
            'alpha1': DeclaredVariableSpec(self.alpha1, ParameterSpec(domain=nonnegative)),
            'alpha2': DeclaredVariableSpec(self.alpha2, ParameterSpec(domain=nonnegative)),
            'delta0': DeclaredVariableSpec(self.delta0, ParameterSpec(domain=nonnegative)),
            'delta1': DeclaredVariableSpec(self.delta1, ParameterSpec(domain=nonnegative)),
            'delta2': DeclaredVariableSpec(self.delta2, ParameterSpec(domain=nonnegative)),
            'amplitude': DeclaredVariableSpec(1.0, ParameterSpec(domain=nonnegative)),
            'period': DeclaredVariableSpec(self.period, ParameterSpec(domain=RealSet.open(0, oo))),
            'width': DeclaredVariableSpec(3.0, ParameterSpec(domain=nonnegative)),
            'stop': DeclaredVariableSpec(self.stop, ParameterSpec()),
        }

    def _get_equations(self) -> Equations:
        """Return the next fast, medium, and slow weights, the fast one driven by the input pulses."""
        return {
            'w0': lambda s: (
                s.target.w0
                + s.target.alpha0 * self._get_input(s) * (s.target.w_max - s.target.w0)
                - s.target.delta0 * (s.target.w0 - s.target.w1)
            ),
            'w1': lambda s: (
                s.target.w1
                + s.target.alpha1 * (s.target.w0 - s.target.w1)
                - s.target.delta1 * (s.target.w1 - s.target.w2)
            ),
            'w2': lambda s: s.target.w2 + s.target.alpha2 * (s.target.w1 - s.target.w2) - s.target.delta2 * s.target.w2,
        }
