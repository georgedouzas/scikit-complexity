"""Define the interaction that trains the parameters of a nested system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ...enums import ElementKind, OptimizerKind
from ...errors import ValidationError
from .._observable_spec import Observable
from .._parameter import _Parameter
from .._role_spec import RoleSpec
from ..laws import BaseLaw, DatasetSpec
from ._interaction import Interaction


class TrainingInteraction(Interaction):
    """An interaction that trains the parameters of a nested system by gradient steps.

    The role `system` binds a nested system, which the training runs over the sequences of a dataset and differentiates.
    One step of the outer system is one epoch: shuffled batches, each followed by one step of the optimizer on the
    parameters named in `trained`. The settings of the optimizer are parameters of the interaction, so a schedule of
    `simulate` can change them between epochs. It states no laws of its own, and `GradientDescentSolver` reads its
    settings. A recurrent neural network learns to predict a time series, an echo state network fits its readout, and a
    learning agent improves its policy between rounds.

    Read more in the user guide, under Interactions.

    Args:
        label:
            The name of the interaction, unique among the components of its system, such as `'training'`.

        system:
            The path of the nested system whose parameters the interaction trains.

        trained:
            The dotted paths of the parameters of the nested system to train, such as `readout.W`.

        loss:
            The observable of the nested system whose weighted mean over the steps of a batch is minimized.

        data:
            The sequences of the parameters of the nested system that set its inputs and targets at every step.

        optimizer:
            The name of the optimizer, one of `OptimizerKind`. `'sgd'` reads the learning rate, `'momentum'` also
            `momentum`, `'rmsprop'` also `beta2`, and `'adam'` also `beta1` and `beta2`.

        learning_rate:
            The learning rate of the optimizer.

        momentum:
            The momentum of `'momentum'`.

        beta1:
            The decay of the mean gradient of `'adam'`.

        beta2:
            The decay of the mean squared gradient of `'adam'` and `'rmsprop'`.

        batch_size:
            The number of sequences of a batch.

        truncation:
            The number of inner steps the gradient flows back through the inner states, or `None` for the whole run. A
            truncation of 1 holds the previous inner states as constant inputs, as Elman's original training does.

        metrics:
            The observables of the nested system recorded at the end of every epoch.

    Raises:
        ValidationError:
            The optimizer is unknown, the batch size or the truncation is not a positive integer, no parameter is
            trained, or the loss, the data, or a metric has the wrong type or label.

    See Also:
        Interaction: Runs its laws once for the elements its roles bind.
        GradientDescentSolver: Runs the epochs of a training.

    Examples:
        >>> from skcomplex.core.components import Observable
        >>> from skcomplex.core.components.interactions import TrainingInteraction
        >>> from skcomplex.core.components.laws import DatasetSpec
        >>> data = DatasetSpec({'inputs.x': [[[0.0]], [[1.0]]]})
        >>> TrainingInteraction('fit', system='net', trained=['W'], loss=Observable('error', 0), data=data).optimizer
        'adam'
    """

    draws = True

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = ()

    def __init__(
        self,
        label: str,
        *,
        system: Annotated[str, RoleSpec(elements=(ElementKind.ANY,))],
        trained: Sequence[str],
        loss: Observable,
        data: DatasetSpec,
        optimizer: str = 'adam',
        learning_rate: Annotated[float, _Parameter(domain=RealSet.closed_open(0, oo))] = 1e-3,
        momentum: Annotated[float, _Parameter(domain=RealSet.closed_open(0, 1))] = 0.9,
        beta1: Annotated[float, _Parameter(domain=RealSet.closed_open(0, 1))] = 0.9,
        beta2: Annotated[float, _Parameter(domain=RealSet.closed_open(0, 1))] = 0.999,
        batch_size: int = 32,
        truncation: int | None = None,
        metrics: Sequence[Observable] = (),
    ) -> None:
        super().__init__(label)
        self.system = system
        self.trained = trained
        self.loss = loss
        self.data = data
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.batch_size = batch_size
        self.truncation = truncation
        self.metrics = metrics

    def _check_counts(self) -> None:
        """Raise an error for an unknown optimizer, or bad batch size or truncation.

        The batch size and the truncation must be positive integers.
        """
        if self.optimizer not in OptimizerKind:
            error_msg = (
                f'The optimizer of {self.label!r} is {self.optimizer!r}, which is not supported. Choose one of '
                f'{", ".join(repr(str(name)) for name in OptimizerKind)}.'
            )
            raise ValidationError(error_msg)
        for name in ('batch_size', 'truncation'):
            value = getattr(self, name)
            if name == 'truncation' and value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give a positive integer.'
                raise ValidationError(error_msg)

    def _check_observables(self) -> None:
        """Raise an error for no trained parameter, bad observables, or a repeated label.

        The loss, the data, and the metrics must be of the right type.
        """
        if isinstance(self.trained, str) or not self.trained or not all(isinstance(p, str) for p in self.trained):
            error_msg = (
                f'{self.label!r} trains {self.trained!r}. Give the dotted paths of the parameters to train, such as '
                "['readout.W']."
            )
            raise ValidationError(error_msg)
        if not isinstance(self.data, DatasetSpec):
            error_msg = f'The data of {self.label!r} is a {type(self.data).__name__}. Give a `DatasetSpec`.'
            raise ValidationError(error_msg)
        observables = [self.loss, *self.metrics]
        if not all(isinstance(item, Observable) for item in observables):
            error_msg = f'The loss and the metrics of {self.label!r} are observables. Give `Observable`s.'
            raise ValidationError(error_msg)
        labels = ['loss', *(item.label for item in self.metrics)]
        if len(set(labels)) < len(labels):
            error_msg = (
                f'The metrics of {self.label!r} are labelled {labels[1:]}. Give distinct labels other than loss.'
            )
            raise ValidationError(error_msg)

    def _check_arguments(self) -> None:
        """Raise an error for an optimizer, counts, parameters, or observables it rejects.

        Raises:
            ValidationError:
                The optimizer is unknown, the batch size or the truncation is not a positive integer, no parameter is
                trained, or the loss, the data, or a metric has the wrong type or label.
        """
        super()._check_arguments()
        self._check_counts()
        self._check_observables()
        self.data._check()
