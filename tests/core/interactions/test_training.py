"""Test the `_training` module of `skcomplex.core.interactions`."""

from types import SimpleNamespace

import numpy as np
import pytest

from skcomplex.core.interactions import OPTIMIZERS, TrainingInteraction
from skcomplex.core.laws import GradientStepLaw
from skcomplex.errors import ValidationError
from skcomplex.specs import DatasetSpec, ObservableSpec

DATA = DatasetSpec({'inputs.x': np.zeros((4, 3, 1))})
LOSS = ObservableSpec('error', 0.0)


def _training(**arguments):
    """Return a training interaction of the readout weights of a nested system `net`."""
    return TrainingInteraction('training', model='net', trained=['readout.W'], loss=LOSS, data=DATA, **arguments)


def test_check_unknown_optimizer():
    """Test that an unknown optimizer raises an error naming it and the supported ones."""
    with pytest.raises(ValidationError, match="'lbfgs-b'") as error:
        _training(optimizer='lbfgs-b').check()
    assert all(repr(name) in str(error.value) for name in OPTIMIZERS)


@pytest.mark.parametrize(('name', 'value'), [('batch_size', 0), ('truncation', 1.5)])
def test_check_counts_that_are_not_positive_integers(name, value):
    """Test that a batch size or a truncation that is not a positive integer raises an error."""
    with pytest.raises(ValidationError, match=name):
        _training(**{name: value}).check()


@pytest.mark.parametrize(
    'arguments',
    [{'trained': []}, {'data': {'inputs.x': np.zeros((4, 3))}}, {'metrics': [ObservableSpec('loss', 0.0)]}],
)
def test_check_trained_data_and_metrics(arguments):
    """Test that no trained parameter, data that is no dataset, or a metric labelled `loss` raise an error."""
    training = TrainingInteraction(
        'training',
        model='net',
        **{'trained': ['readout.W'], 'loss': LOSS, 'data': DATA, **arguments},
    )
    with pytest.raises(ValidationError):
        training.check()


def test_run_returns_the_gradient_step_law():
    """Test that the training returns one gradient-step law with its arguments."""
    laws = _training(truncation=1).check().prepare().run(SimpleNamespace()).laws_
    assert len(laws) == 1
    assert isinstance(laws[0], GradientStepLaw)
    assert (laws[0].trained, laws[0].truncation, laws[0].optimizer) == (['readout.W'], 1, 'adam')


def test_get_params_lists_the_hyperparameters():
    """Test that the hyperparameters of the optimizer are parameters of the interaction."""
    assert {'learning_rate', 'momentum', 'beta1', 'beta2'} <= set(_training().get_params())
