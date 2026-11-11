"""Neural network systems: Elman's simple recurrent network.

It is trained on the temporal XOR stream.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import exp

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import CouplingInteraction, TrainingInteraction
from ....core.components.spaces import DiscreteTime
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import (
    DatasetSpec,
    ObservableSpec,
    ParameterSpec,
    PhaseSpec,
    PhasesScheduleSpec,
    RoleSpec,
    StateSpec,
    VisualSpec,
)
from ....core.types import Array

TRAINED = ('input.D', 'recurrence.A', 'readout.W', 'hidden.b', 'outputs.c')
WEIGHT_SCALE = 0.5
PHASES = (PhaseSpec('drive'), PhaseSpec('hidden'), PhaseSpec('readout'))


class HiddenUnitElement(Element):
    """Hidden unit of a simple recurrent network.

    Its activity is a sigmoid of its recurrent input and its drive. The drive `u` holds the weighted input of the
    previous step, and the bias `b` is adaptive, so training changes it.

    Args:
        label:
            The label of the element or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        s:
            The activity in `(0, 1)`.

        u:
            The weighted input of the previous step.

        b:
            The bias, an adaptive parameter.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import HiddenUnitElement
        >>> HiddenUnitElement('unit').s
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        s: Annotated[float, StateSpec()] = 0.5,
        u: Annotated[float, StateSpec()] = 0.0,
        b: Annotated[float, ParameterSpec(adaptive=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.s = s
        self.u = u
        self.b = b


class InputUnitElement(Element):
    """Input unit of a simple recurrent network.

    A dataset or a schedule sets its value at every step.

    Args:
        label:
            The label of the element or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The input.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import InputUnitElement
        >>> InputUnitElement('bit').x
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x


class OutputUnitElement(Element):
    """Output unit of a simple recurrent network, whose activity predicts a target.

    The bias `c` is adaptive, so training changes it, and the target is a parameter a dataset or a schedule sets.

    Args:
        label:
            The label of the element or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        y:
            The prediction in `(0, 1)`.

        c:
            The bias, an adaptive parameter.

        target:
            The value the prediction is compared with.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import OutputUnitElement
        >>> OutputUnitElement('prediction').target
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        y: Annotated[float, StateSpec()] = 0.5,
        c: Annotated[float, ParameterSpec(adaptive=True)] = 0.0,
        target: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.y = y
        self.c = c
        self.target = target


class InputCouplingInteraction(CouplingInteraction):
    """Weighted input of a simple recurrent network.

    It sets the drive `u = D x` of every hidden unit each step.

    The weights `D` are an adaptive matrix with one row per hidden unit and one column per input unit, so training
    changes them. The coupling runs in the phase `drive` of the step, which the system declares before the phases
    `hidden` of the recurrence and `readout` of the readout, so the hidden units read the drive of the same step.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of hidden units.

        source:
            The path of the group of input units.

        D:
            The input weights, a matrix of shape `(hidden units, input units)` or a number for every entry.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import InputCouplingInteraction
        >>> InputCouplingInteraction('input', target='hidden', source='inputs').write
        'u'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
        D: Annotated[object, ParameterSpec(shape=('target', 'source'), adaptive=True)] = 0.0,
    ) -> None:
        super().__init__(
            label,
            target=target,
            source=source,
            read='x',
            write='u',
            function=self._get_input,
            mode='group',
            weights='D',
            form='map',
            phase='drive',
        )
        self.D = D

    def _get_input(self, namespace: SimpleNamespace) -> object:
        """Return the input of the other unit."""
        return namespace.other.x


class ReadoutCouplingInteraction(CouplingInteraction):
    """Readout of a simple recurrent network, `y_i <- sigmoid(sum_j W_ij s_j + c_i)`.

    The sigmoid is the logistic one. Every output unit reads the activities of the hidden units through the adaptive
    matrix `W`, adds its bias `c`, and takes the sigmoid of the sum as its next prediction.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of output units.

        source:
            The path of the group of hidden units.

        W:
            The readout weights, a matrix of shape `(output units, hidden units)` or a number for every entry.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import ReadoutCouplingInteraction
        >>> ReadoutCouplingInteraction('readout', target='outputs', source='hidden').write
        'y'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
        W: Annotated[object, ParameterSpec(shape=('target', 'source'), adaptive=True)] = 0.0,
    ) -> None:
        super().__init__(
            label,
            target=target,
            source=source,
            read='s',
            write='y',
            function=self._get_activity,
            mode='group',
            weights='W',
            form='map',
            phase='readout',
            combine=self._activate,
        )
        self.W = W

    def _get_activity(self, namespace: SimpleNamespace) -> object:
        """Return the activity of the hidden unit."""
        return namespace.other.s

    def _activate(self, namespace: SimpleNamespace, reduction: object) -> object:
        """Return the sigmoid of the weighted activities and the bias of the output unit."""
        return 1 / (1 + exp(-(reduction + namespace.member.c)))


class RecurrenceCouplingInteraction(CouplingInteraction):
    """Recurrence of a simple recurrent network.

    It sets `s_i <- sigmoid(sum_j A_ij s_j + u_i + b_i)`. The sigmoid is the logistic one. Every hidden unit reads the
    activities of every hidden unit, itself included, through the adaptive matrix `A`, adds its drive `u` and its bias
    `b`, and takes the sigmoid of the sum as its next activity (Gros 2015 Eq. 10.18).

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of hidden units.

        A:
            The recurrent weights, a square matrix with one row and one column per hidden unit, or a number for every
            entry.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import RecurrenceCouplingInteraction
        >>> RecurrenceCouplingInteraction('recurrence', target='hidden').include_self
        True
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        A: Annotated[object, ParameterSpec(shape=('target', 'target'), adaptive=True)] = 0.0,
    ) -> None:
        super().__init__(
            label,
            target=target,
            read='s',
            write='s',
            function=self._get_activity,
            mode='group',
            weights='A',
            include_self=True,
            form='map',
            phase='hidden',
            combine=self._activate,
        )
        self.A = A

    def _get_activity(self, namespace: SimpleNamespace) -> object:
        """Return the activity of the other unit."""
        return namespace.other.s

    def _activate(self, namespace: SimpleNamespace, reduction: object) -> object:
        """Return the sigmoid of the recurrent input, the drive, and the bias of the unit."""
        return 1 / (1 + exp(-(reduction + namespace.member.u + namespace.member.b)))


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
    ) -> None:
        super().__init__(label)
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

    def _build_system(self) -> CompositeSystem:
        """Return Elman's simple recurrent network.

        It learns to predict the next bit of a temporal XOR stream.
        """
        rng = np.random.default_rng(self.random_state)
        data = self._get_dataset(self._get_temporal_xor(self.n_bits, rng), self.window)
        net = CompositeSystem(
            'net',
            elements=[
                GroupElement('inputs', element=InputUnitElement, size=1),
                GroupElement('hidden', element=HiddenUnitElement, size=self.n_hidden),
                GroupElement('outputs', element=OutputUnitElement, size=1),
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
            time=DiscreteTime(schedule=PhasesScheduleSpec(PHASES)),
        )
        training = TrainingInteraction(
            'training',
            system='net',
            trained=TRAINED,
            loss=ObservableSpec('error', group='outputs', reduce='sum', expression=self._get_squared_error),
            data=data,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            truncation=self.truncation,
            metrics=(
                ObservableSpec('xor_error', group='outputs', reduce='sum', expression=self._get_squared_error),
                ObservableSpec('random_error', group='outputs', reduce='sum', expression=self._get_squared_error),
            ),
        )
        return CompositeSystem(self.label, systems=[net], interactions=[training], time=DiscreteTime())
