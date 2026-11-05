"""Test the `_morris` module of `skcomplex.experiments`."""

import numpy as np

from skcomplex.experiments import MorrisDesign


def test_morris_design_moves_one_name_per_step():
    """Test that every step of a Morris trajectory changes exactly one name and stays within the bounds."""
    rows = MorrisDesign(n=5, random_state=0).prepare({'a': (0.0, 1.0), 'b': (10.0, 20.0)}).run().rows_
    values = np.asarray([[row['a'], row['b']] for row in rows]).reshape(5, 3, 2)
    assert ((np.diff(values, axis=1) != 0).sum(axis=2) == 1).all()
    assert values[..., 0].min() >= 0.0
    assert values[..., 1].max() <= 20.0
