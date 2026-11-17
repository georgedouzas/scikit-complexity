"""Neuroscience systems: adaptive synapses with memory on several time scales."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ....core.components.laws.utils import mod, where
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
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
        >>> AdaptrodeSystem(stop=50.0).prepare().system_.elements[0].values['stop']
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
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

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state by name."""
        return {
            'w0': 0.0,
            'w1': 0.0,
            'w2': 0.0,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the value of every parameter by name."""
        return {
            'w_max': 1.0,
            'alpha0': self.alpha0,
            'alpha1': self.alpha1,
            'alpha2': self.alpha2,
            'delta0': self.delta0,
            'delta1': self.delta1,
            'delta2': self.delta2,
            'amplitude': 1.0,
            'period': self.period,
            'width': 3.0,
            'stop': self.stop,
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
