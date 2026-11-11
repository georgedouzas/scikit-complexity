"""Define the interaction that trains the adaptive parameters of a nested system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import DatasetSpec, ObservableSpec, ParameterSpec, RoleSpec
from ...types import ElementKind
from ..base import BaseComponent
from ..laws import BaseLaw, GradientStepLaw
from ..spaces import BaseSpace
from ._interaction import Interaction

OPTIMIZERS = ('sgd', 'momentum', 'rmsprop', 'adam')


class TrainingInteraction(Interaction):
    """An interaction that trains the parameters of a nested system by gradient steps.

    The role `system` binds a nested system, which the training runs over the sequences of a dataset from its initial
    states and differentiates. One step of the outer system is one epoch: shuffled batches, each followed by one update
    of the optimizer on the adaptive parameters named in `trained`. The hyperparameters of the optimizer are parameters
    of the interaction, so a schedule of `simulate` can change them between epochs. The `optimizer` reads them as
    follows:

    - `'sgd'`: the learning rate.
    - `'momentum'`: the learning rate and `momentum`.
    - `'rmsprop'`: the learning rate and `beta2`, the decay of its mean squared gradient.
    - `'adam'`: the learning rate, `beta1`, and `beta2`.

    Unlike `GradientInteraction`, which moves states down a potential in the time of the system, it fits parameters of
    another system to data, one epoch per step. Choose it when a component of the model learns from data during the run.

    It appears in many fields. A recurrent neural network learns to predict a time series, an echo state network fits
    its readout, and a learning agent improves its policy between rounds.

    Args:
        label:
            The name of the interaction, unique among the components of its system, such as `'training'`.

        system:
            The path of the nested system whose parameters the interaction trains.

        trained:
            The dotted paths of the adaptive parameters of the nested system to train, such as `readout.W`.

        loss:
            The observable of the nested system whose weighted mean over the steps of a batch is minimized.

        data:
            The sequences of the parameters of the nested system that set its inputs and targets at every step.

        optimizer:
            The name of the optimizer, one of `OPTIMIZERS`.

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
            The number of inner steps the gradient flows back through the inner states, or `None` for the whole run.
            A truncation of 1 holds the previous inner states as constant inputs, as Elman's original training does.

        metrics:
            The observables of the nested system recorded at the end of every epoch.

    Raises:
        ValidationError:
            The optimizer is unknown, the batch size or the truncation is not a positive integer, no parameter is
            trained, or the loss, the data, or a metric has the wrong type or label.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.components.interactions import TrainingInteraction
        >>> from skcomplex.core.specs import DatasetSpec, ObservableSpec
        >>> data = DatasetSpec({'inputs.x': np.zeros((4, 3, 1))})
        >>> training = TrainingInteraction(
        ...     'training', system='net', trained=['readout.W'], loss=ObservableSpec('error', 0.0), data=data
        ... )
        >>> training.optimizer
        'adam'
    """

    draws = True

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (GradientStepLaw,)

    def __init__(
        self,
        label: str,
        *,
        system: Annotated[str, RoleSpec(elements=(ElementKind.ANY,))],
        trained: Sequence[str],
        loss: ObservableSpec,
        data: DatasetSpec,
        optimizer: str = 'adam',
        learning_rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1e-3,
        momentum: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, 1))] = 0.9,
        beta1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, 1))] = 0.9,
        beta2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, 1))] = 0.999,
        batch_size: int = 32,
        truncation: int | None = None,
        metrics: Sequence[ObservableSpec] = (),
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
        if self.optimizer not in OPTIMIZERS:
            error_msg = (
                f'The optimizer of {self.label!r} is {self.optimizer!r}, which is not supported. Choose one of '
                f'{", ".join(map(repr, OPTIMIZERS))}.'
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
            error_msg = f'{self.label!r} trains {self.trained!r}. Give the dotted paths of adaptive parameters.'
            raise ValidationError(error_msg)
        if not isinstance(self.data, DatasetSpec):
            error_msg = f'The data of {self.label!r} is a {type(self.data).__name__}. Give a `DatasetSpec`.'
            raise ValidationError(error_msg)
        observables = [self.loss, *self.metrics]
        if not all(isinstance(item, ObservableSpec) for item in observables):
            error_msg = f'The loss and the metrics of {self.label!r} are observables. Give `ObservableSpec`s.'
            raise ValidationError(error_msg)
        labels = ['loss', *(item.label for item in self.metrics)]
        if len(set(labels)) < len(labels):
            error_msg = (
                f'The metrics of {self.label!r} are labelled {labels[1:]}. Give distinct labels other than loss.'
            )
            raise ValidationError(error_msg)

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for an optimizer, counts, parameters, or observables it rejects.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The optimizer is unknown, the batch size or the truncation is not a positive integer, no parameter is
                trained, or the loss, the data, or a metric has the wrong type or label.
        """
        super()._prepare(context)
        self._check_counts()
        self._check_observables()
        self.data._check()

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseComponent]:
        """Return the gradient-step law of the training."""
        del namespace
        law = GradientStepLaw(
            self.trained,
            self.loss,
            self.data,
            self.optimizer,
            self.batch_size,
            truncation=self.truncation,
            metrics=self.metrics,
            label=self.label,
        )
        return [law]
