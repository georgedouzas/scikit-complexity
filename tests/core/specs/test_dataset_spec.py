"""Test the `_dataset_spec` module of `skcomplex.core.specs`."""

import numpy as np
import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import DatasetSpec


def test_dataset_spec_counts_sequences_and_steps():
    """Test that a dataset counts the sequences and steps of its first array."""
    data = DatasetSpec({'inputs.x': np.zeros((5, 12, 1)), 'outputs.target': np.zeros((5, 12, 1))})
    assert (data.sequences, data.steps) == (5, 12)


@pytest.mark.parametrize('values', [{}, {'inputs.x': np.zeros(5)}, {1: np.zeros((5, 12))}])
def test_dataset_spec_rejects_values_it_cannot_take(values):
    """Test that empty values, an array of one dimension, or a path that is not a string raise an error."""
    with pytest.raises(ValidationError):
        DatasetSpec(values)._check()
