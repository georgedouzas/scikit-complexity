"""Neural network systems: Elman's simple recurrent network.

It is trained on the temporal XOR stream.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import exp

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction, TrainingInteraction
from ....core.components.laws import DatasetSpec, UpdateLaw
from ....core.components.laws.utils import total
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role

TRAINED = ('input.D', 'recurrence.A', 'readout.W', 'hidden.b', 'outputs.c')
WEIGHT_SCALE = 0.5


class HiddenUnitElement(Element):
    """Hidden units of a simple recurrent network.

    The activity of a unit is a sigmoid of its recurrent input and its drive. The drive `u` holds the weighted input
    of the step, and the training sets the bias `b`.

    Args:
        label:
            The label of the units.

        s:
            The activity in `(0, 1)`.

        u:
            The weighted input of the step.

        b:
            The bias.

        size:
            The number of units, or `None` for one.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import HiddenUnitElement
        >>> HiddenUnitElement('unit').s
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        s: float = 0.5,
        u: float = 0.0,
        b: float = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.s = s
        self.u = u
        self.b = b


class InputUnitElement(Element):
    """Input units of a simple recurrent network.

    A dataset or a schedule sets the value `x` at every step.

    Args:
        label:
            The label of the units.

        x:
            The input.

        size:
            The number of units, or `None` for one.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import InputUnitElement
        >>> InputUnitElement('bit').x
        0.0
    """

    def __init__(self, label: str, *, x: float = 0.0, size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.x = x


class OutputUnitElement(Element):
    """Output units of a simple recurrent network, whose activity predicts a target.

    The training sets the bias `c`, and a dataset or a schedule sets the target.

    Args:
        label:
            The label of the units.

        y:
            The prediction in `(0, 1)`.

        c:
            The bias.

        target:
            The value the prediction is compared with.

        size:
            The number of units, or `None` for one.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import OutputUnitElement
        >>> OutputUnitElement('prediction').target
        0.0
    """

    def __init__(
        self,
        label: str,
        *,
        y: float = 0.5,
        c: float = 0.0,
        target: float = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.y = y
        self.c = c
        self.target = target


class InputCouplingInteraction(Interaction):
    """Weighted input of a simple recurrent network, `u_i <- sum_j D_ij x_j`.

    It sets the drive of every hidden unit at each step from the input units. The weights `D` have one row per hidden
    unit and one column per input unit. The coupling runs in the phase `drive` of the step, before the phases `hidden`
    of the recurrence and `readout` of the readout, so the hidden units read the drive of the same step.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of hidden units.

        source:
            The path of the group of input units.

        D:
            The input weights, a matrix of shape `(hidden units, input units)`.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import InputCouplingInteraction
        >>> InputCouplingInteraction('input', target='hidden', source='inputs').source
        'inputs'
    """

    def __init__(self, label: str, *, target: Role, source: Role, D: object = 0.0) -> None:
        super().__init__(label)
        self.target = target
        self.source = source
        self.D = D

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the drive of the hidden units in the phase `drive`.

        Args:
            context:
                The namespace of the units, the time `t`, and the parameters.
        """
        super()._prepare(context)
        drive = total(context.source.x, weights=context.D)
        self.laws_: list[BaseComponent] = [UpdateLaw(context.target.u, drive, phase='drive')]


class RecurrenceCouplingInteraction(Interaction):
    """Recurrence of a simple recurrent network, `s_i <- sigmoid(sum_j A_ij s_j + u_i + b_i)`.

    The sigmoid is the logistic one. Every hidden unit reads the activities of every hidden unit, itself included,
    through the matrix `A`, adds its drive `u` and its bias `b`, and takes the sigmoid of the sum as its next activity
    (Gros 2015 Eq. 10.18), in the phase `hidden` of the step.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of hidden units.

        A:
            The recurrent weights, a square matrix with one row and one column per hidden unit.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import RecurrenceCouplingInteraction
        >>> RecurrenceCouplingInteraction('recurrence', target='hidden').target
        'hidden'
    """

    def __init__(self, label: str, *, target: Role, A: object = 0.0) -> None:
        super().__init__(label)
        self.target = target
        self.A = A

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the next activity of the hidden units in the phase `hidden`.

        Args:
            context:
                The namespace of the units, the time `t`, and the parameters.
        """
        super()._prepare(context)
        hidden = context.target
        recurrent = total(hidden.s, weights=context.A)
        activity = 1 / (1 + exp(-(recurrent + hidden.u + hidden.b)))
        self.laws_: list[BaseComponent] = [UpdateLaw(hidden.s, activity, phase='hidden')]


class ReadoutCouplingInteraction(Interaction):
    """Readout of a simple recurrent network, `y_i <- sigmoid(sum_j W_ij s_j + c_i)`.

    The sigmoid is the logistic one. Every output unit reads the activities of the hidden units through the matrix
    `W`, adds its bias `c`, and takes the sigmoid of the sum as its next prediction, in the phase `readout` of the step.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of output units.

        source:
            The path of the group of hidden units.

        W:
            The readout weights, a matrix of shape `(output units, hidden units)`.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import ReadoutCouplingInteraction
        >>> ReadoutCouplingInteraction('readout', target='outputs', source='hidden').source
        'hidden'
    """

    def __init__(self, label: str, *, target: Role, source: Role, W: object = 0.0) -> None:
        super().__init__(label)
        self.target = target
        self.source = source
        self.W = W

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the next prediction of the output units in the phase `readout`.

        Args:
            context:
                The namespace of the units, the time `t`, and the parameters.
        """
        super()._prepare(context)
        outputs = context.target
        readout = total(context.source.s, weights=context.W)
        prediction = 1 / (1 + exp(-(readout + outputs.c)))
        self.laws_: list[BaseComponent] = [UpdateLaw(outputs.y, prediction, phase='readout')]


class ElmanNetworkSystem(BaseSystem):
    """Elman's simple recurrent network.

    It learns to predict the next bit of a temporal XOR stream. The nested discrete map `net` holds one input unit,
    `n_hidden` hidden units, and one output unit. The input sets the drive `u = D x` of the hidden units, the hidden
    units take `s <- sigmoid(A s + u + b)` (Gros 2015 Eq. 10.18), and the output predicts `y <- sigmoid(W s + c)`, in
    three phases of every step, so the prediction of a step reads its input. The training interaction `training` trains
    `D`, `A`, `W`, `b`, and `c` with Adam on the squared error of predicting the next input from the inputs so far, over
    windows of `window` bits that start at every third bit of a stream in which every third bit is the exclusive or of
    the two before it. Only those bits can be predicted, so the metric `xor_error` falls toward 0 while `random_error`,
    the error on the other bits, stays near 0.25. One step of the system is one epoch. The system has the nested system
    `net` and the training interaction `training`.

    Args:
        n_hidden:
            The number of hidden units.

        n_bits:
            The length of the stream.

        window:
            The number of bits of a sequence.

        batch_size:
            The number of sequences of a batch.

        learning_rate:
            The learning rate of Adam.

        truncation:
            The number of steps the gradient flows back through the hidden activities, 1 for Elman's original rule
            that holds the previous activities as constant context, or `None` for full backpropagation through time.

        random_state:
            The seed or generator of the stream and of the initial weights.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import ElmanNetworkSystem
        >>> [item.label for item in ElmanNetworkSystem().prepare().system_.systems]
        ['net']
    """

    field = 'computer_science/neural_networks'
    topics = ('neural network', 'recurrent network', 'learning', 'backpropagation through time', 'discrete map')
    references = ('Gros 2015 §10.5.1', 'Elman 1990')
    until = 200
    dt = 1

    def __init__(
        self,
        n_hidden: int = 4,
        *,
        n_bits: int = 3000,
        window: int = 12,
        batch_size: int = 16,
        learning_rate: float = 0.05,
        truncation: int | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_hidden = n_hidden
        self.n_bits = n_bits
        self.window = window
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.truncation = truncation
        self.random_state = random_state

    @staticmethod
    def _get_squared_error(member: SimpleNamespace) -> object:
        """Return the squared error of the prediction of an output unit."""
        return (member.y - member.target) ** 2

    @staticmethod
    def _get_temporal_xor(n_bits: int, rng: np.random.Generator) -> Array:
        """Return a stream of bits with a temporal XOR.

        Every third bit is the exclusive or of the two bits before it.
        """
        bits = rng.integers(0, 2, n_bits).astype(float)
        triples = n_bits // 3
        bits[2 : 3 * triples : 3] = np.logical_xor(bits[0 : 3 * triples : 3], bits[1 : 3 * triples : 3])
        return bits

    @staticmethod
    def _get_dataset(bits: Array, window: int) -> DatasetSpec:
        """Return the windows of a stream, the next bits as targets, and the metric masks.

        The windows start at every third bit. The prediction of a step reads the input of that step, so the target of
        step `k` is the input of step `k + 1`.
        """
        starts = np.arange(0, len(bits) - window, 3)
        inputs = np.stack([bits[start : start + window] for start in starts])
        targets = np.stack([bits[start + 1 : start + window + 1] for start in starts])
        xor = np.broadcast_to(np.arange(window) % 3 == 1, inputs.shape).astype(float)
        return DatasetSpec(
            {'inputs.x': inputs[..., np.newaxis], 'outputs.target': targets[..., np.newaxis]},
            weights={'xor_error': xor, 'random_error': 1.0 - xor},
        )

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the network and the training that teaches it the temporal XOR stream."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        data = self._get_dataset(self._get_temporal_xor(self.n_bits, rng), self.window)
        net = System(
            'net',
            elements=[
                InputUnitElement('inputs', size=1),
                HiddenUnitElement('hidden', size=self.n_hidden),
                OutputUnitElement('outputs', size=1),
            ],
            interactions=[
                InputCouplingInteraction(
                    'input',
                    target='hidden',
                    source='inputs',
                    D=rng.normal(0.0, WEIGHT_SCALE, (self.n_hidden, 1)),
                ),
                RecurrenceCouplingInteraction(
                    'recurrence',
                    target='hidden',
                    A=rng.normal(0.0, WEIGHT_SCALE, (self.n_hidden, self.n_hidden)),
                ),
                ReadoutCouplingInteraction(
                    'readout',
                    target='outputs',
                    source='hidden',
                    W=rng.normal(0.0, WEIGHT_SCALE, (1, self.n_hidden)),
                ),
            ],
        )
        training = TrainingInteraction(
            'training',
            system='net',
            trained=TRAINED,
            loss=Observable('error', group='outputs', reduce='sum', expression=self._get_squared_error),
            data=data,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            truncation=self.truncation,
            metrics=(
                Observable('xor_error', group='outputs', reduce='sum', expression=self._get_squared_error),
                Observable('random_error', group='outputs', reduce='sum', expression=self._get_squared_error),
            ),
        )
        self.system_ = System(self.label, systems=[net], interactions=[training])
