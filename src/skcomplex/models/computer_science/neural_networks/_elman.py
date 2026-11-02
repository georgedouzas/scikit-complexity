"""Neural network models: Elman's simple recurrent network, trained on the temporal XOR stream."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import numpy as np

from ....core.elements import GroupElement
from ....core.interactions import TrainingInteraction
from ....core.spaces import DiscreteTime
from ....core.system import System
from ....specs import DatasetSpec, ObservableSpec, PhaseSpec, PhasesScheduleSpec
from ....types import Array
from ..._catalog import register_model
from ._hidden_unit import HiddenUnitElement
from ._input_coupling import InputCouplingInteraction
from ._input_unit import InputUnitElement
from ._output_unit import OutputUnitElement
from ._readout_coupling import ReadoutCouplingInteraction
from ._recurrence_coupling import RecurrenceCouplingInteraction

TRAINED = ('input.D', 'recurrence.A', 'readout.W', 'hidden.b', 'outputs.c')
WEIGHT_SCALE = 0.5
PHASES = (PhaseSpec('drive'), PhaseSpec('hidden'), PhaseSpec('readout'))


def _squared_error(member: SimpleNamespace) -> object:
    """Return the squared error of the prediction of an output unit."""
    return (member.y - member.target) ** 2


def _temporal_xor(n_bits: int, rng: np.random.Generator) -> Array:
    """Return a stream of bits whose every third bit is the exclusive or of the two bits before it."""
    bits = rng.integers(0, 2, n_bits).astype(float)
    triples = n_bits // 3
    bits[2 : 3 * triples : 3] = np.logical_xor(bits[0 : 3 * triples : 3], bits[1 : 3 * triples : 3])
    return bits


def _dataset(bits: Array, window: int) -> DatasetSpec:
    """Return the windows of a stream that start at every third bit, the next bits as targets, and the metric masks.

    The prediction of a step reads the input of that step, so the target of step `k` is the input of step `k + 1`.
    """
    starts = np.arange(0, len(bits) - window, 3)
    inputs = np.stack([bits[start : start + window] for start in starts])
    targets = np.stack([bits[start + 1 : start + window + 1] for start in starts])
    xor = np.broadcast_to(np.arange(window) % 3 == 1, inputs.shape).astype(float)
    return DatasetSpec(
        {'inputs.x': inputs[..., np.newaxis], 'outputs.target': targets[..., np.newaxis]},
        weights={'xor_error': xor, 'random_error': 1.0 - xor},
    )


@register_model(
    field='computer_science/neural_networks',
    tags=('neural network', 'recurrent network', 'learning', 'backpropagation through time', 'discrete map'),
    references=('Gros 2015 §10.5.1', 'Elman 1990'),
    defaults={'until': 200, 'dt': 1},
)
def elman_network(
    n_hidden: int = 4,
    *,
    n_bits: int = 3000,
    window: int = 12,
    batch_size: int = 16,
    learning_rate: float = 0.05,
    truncation: int | None = None,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Elman's simple recurrent network, which learns to predict the next bit of a temporal XOR stream.

    The nested discrete map `net` holds one input unit, `n_hidden` hidden units, and one output unit. The input sets
    the drive `u = D x` of the hidden units, the hidden units take `s <- sigmoid(A s + u + b)` (Gros 2015 Eq. 10.18),
    and the output predicts `y <- sigmoid(W s + c)`, in three phases of every step, so the prediction of a step reads
    its input. The training interaction `training` trains `D`, `A`, `W`, `b`, and `c` with Adam on the squared error
    of predicting the next input from the inputs so far, over windows of `window` bits that start at every third bit of
    a stream in which every third bit is the exclusive or of the two before it. Only those bits can be predicted, so
    the metric `xor_error` falls toward 0 while `random_error`, the error on the other bits, stays near 0.25. One step
    of the system is one epoch.

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

    Returns:
        system:
            The model, with the nested system `net` and the training interaction `training`.

    Examples:
        >>> from skcomplex.models.computer_science.neural_networks import elman_network
        >>> [item.label for item in elman_network().systems]
        ['net']
    """
    rng = np.random.default_rng(random_state)
    data = _dataset(_temporal_xor(n_bits, rng), window)
    net = System(
        'net',
        elements=[
            GroupElement('inputs', element=InputUnitElement, size=1),
            GroupElement('hidden', element=HiddenUnitElement, size=n_hidden),
            GroupElement('outputs', element=OutputUnitElement, size=1),
        ],
        interactions=[
            InputCouplingInteraction(
                'input',
                target='hidden',
                source='inputs',
                D=rng.normal(0.0, WEIGHT_SCALE, (n_hidden, 1)),
            ),
            RecurrenceCouplingInteraction(
                'recurrence',
                target='hidden',
                A=rng.normal(0.0, WEIGHT_SCALE, (n_hidden, n_hidden)),
            ),
            ReadoutCouplingInteraction(
                'readout',
                target='outputs',
                source='hidden',
                W=rng.normal(0.0, WEIGHT_SCALE, (1, n_hidden)),
            ),
        ],
        time=DiscreteTime(schedule=PhasesScheduleSpec(PHASES)),
    )
    training = TrainingInteraction(
        'training',
        model='net',
        trained=TRAINED,
        loss=ObservableSpec('error', group='outputs', reduce='sum', expression=_squared_error),
        data=data,
        learning_rate=learning_rate,
        batch_size=batch_size,
        truncation=truncation,
        metrics=(
            ObservableSpec('xor_error', group='outputs', reduce='sum', expression=_squared_error),
            ObservableSpec('random_error', group='outputs', reduce='sum', expression=_squared_error),
        ),
    )
    return System('elman', systems=[net], interactions=[training], time=DiscreteTime())
