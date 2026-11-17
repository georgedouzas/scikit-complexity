"""Test the `_hilbert_spec` module of `skcomplex.systems.physics.common.quantum`."""

import numpy as np
import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.systems.physics.common.quantum import HilbertSpec, QuantumCompiler


@pytest.mark.parametrize(
    ('spec', 'message'),
    [
        (HilbertSpec(0), 'positive integer'),
        (HilbertSpec(True), 'positive integer'),
        (HilbertSpec(2, {'z': (np.eye(3), True)}), 'shape'),
        (HilbertSpec(2, basis={'up': 2}), 'outside the dimension'),
        (HilbertSpec(2, {'x': (np.eye(2), True)}), "display operator 'z'"),
    ],
)
def test_hilbert_spec_checks_its_space(spec, message):
    """Test that a Hilbert space needs a positive dimension, square operators, basis states, and its display."""
    with pytest.raises(ValidationError, match=message):
        spec._check()


def test_hilbert_spec_names_the_quantum_compiler():
    """Test that a Hilbert space names the compiler of quantum systems."""
    assert isinstance(HilbertSpec(2)._get_compiler(), QuantumCompiler)
