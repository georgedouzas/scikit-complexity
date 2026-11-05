"""Define gradient-step laws, optimizer steps on the adaptive parameters of a nested system once per epoch."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ...specs import DatasetSpec, ObservableSpec, RequirementSpec
from ..spaces import DiscreteTime
from ._base import BaseLaw


class GradientStepLaw(BaseLaw):
    """Optimizer steps on adaptive parameters of a nested system, on the gradient of a loss over unrolled runs.

    One step of the discrete time that holds the law is one epoch. In an epoch the sequences of the dataset are shuffled
    and split into batches, and for every batch the nested system runs from its initial states over every sequence,
    the loss is averaged over the steps and sequences with the step weights, and the optimizer changes the trained
    parameters by the gradient of that mean. The gradient is exact, by automatic differentiation of the unrolled runs.
    The law records the mean batch loss of the epoch and every metric over the whole dataset at the end of the epoch.

    Args:
        trained:
            The dotted paths of the adaptive parameters of the nested system that the law trains, such as
            `readout.W`.

        loss:
            The observable of the nested system whose weighted mean over the steps of a batch is minimized.

        data:
            The sequences of the parameters of the nested system that set its inputs and targets at every step.

        optimizer:
            The name of the optimizer, `'sgd'`, `'momentum'`, `'rmsprop'`, or `'adam'`.

        batch_size:
            The number of sequences of a batch. The last batch of an epoch holds the rest.

        truncation:
            The number of inner steps the gradient flows back through the inner states, or `None` for the whole run.

        metrics:
            The observables of the nested system recorded at the end of every epoch, as weighted means over every
            sequence.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.laws import GradientStepLaw
        >>> from skcomplex.specs import DatasetSpec, ObservableSpec
        >>> data = DatasetSpec({'inputs.x': np.zeros((4, 3, 1))})
        >>> law = GradientStepLaw(['readout.W'], ObservableSpec('error', 0.0), data, 'adam', 2)
        >>> law.prepare('training').targets_
        ()
    """

    def __init__(
        self,
        trained: Sequence[str],
        loss: ObservableSpec,
        data: DatasetSpec,
        optimizer: str,
        batch_size: int,
        *,
        truncation: int | None = None,
        metrics: Sequence[ObservableSpec] = (),
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.trained = trained
        self.loss = loss
        self.data = data
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.truncation = truncation
        self.metrics = metrics

    def _get_targets(self) -> tuple[object, ...]:
        """Return no state, since the law changes parameters of a nested system."""
        return ()

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: discrete time, one step per epoch."""
        return RequirementSpec(time=(DiscreteTime,))

    def _check(self) -> None:
        """Accept the arguments, which the interaction that returns the law validates."""
