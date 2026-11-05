"""Test the tensor spec."""

import numpy as np
import pytest

from skcomplex.errors import ValidationError
from skcomplex.specs import TensorSpec


@pytest.mark.parametrize(
    ('symmetry', 'n_dim', 'count'),
    [
        ('full', 2, 4),
        ('full', 3, 9),
        ('symmetric', 2, 3),
        ('symmetric', 3, 6),
        ('antisymmetric', 2, 1),
        ('antisymmetric', 3, 3),
        ('traceless', 2, 2),
        ('traceless', 3, 5),
    ],
)
def test_indices_count_the_independent_components(symmetry, n_dim, count):
    """Every symmetry has the number of independent components of its dimension."""
    assert len(TensorSpec(symmetry, n_dim).indices) == count


def test_check_rejects_one_dimension():
    """A tensor in one dimension is not valid."""
    with pytest.raises(ValidationError, match='2 or 3 dimensions'):
        TensorSpec('symmetric', 1).prepare()


def test_array_of_gives_a_traceless_tensor():
    """The tensors of random components are symmetric and traceless, on the last two axes."""
    components = np.random.default_rng(0).normal(size=(4, 5))
    tensors = TensorSpec('traceless', 3).array_of(components)
    assert tensors.shape == (4, 3, 3)
    assert np.allclose(np.trace(tensors, axis1=1, axis2=2), 0.0)
    assert np.allclose(tensors, np.swapaxes(tensors, 1, 2))


def test_components_of_rejects_a_matrix_of_another_symmetry():
    """An asymmetric matrix has no components of a symmetric tensor."""
    with pytest.raises(ValidationError, match='symmetry'):
        TensorSpec('symmetric', 2).components_of([[0.0, 1.0], [2.0, 0.0]])
