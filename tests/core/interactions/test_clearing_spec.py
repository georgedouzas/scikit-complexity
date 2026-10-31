"""Test the `_clearing_spec` module of `skcomplex.core.interactions`."""

import numpy as np

from skcomplex.core.interactions import ClearingSpec


def test_check_keeps_the_storage():
    """Test that a checked clearing keeps the storage that persists across clearings."""
    storage = {'price': 2.0}
    assert ClearingSpec([], [], np.random.default_rng(0), storage).check().storage is storage
