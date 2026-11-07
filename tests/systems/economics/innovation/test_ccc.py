"""Test the `_ccc` module of `skcomplex.systems.economics.innovation`."""

import numpy as np

from skcomplex.systems.economics.innovation import ccc_model

from ..conftest import _run


def test_ccc_model_is_punctuated_between_full_and_empty_regimes():
    """Test punctuated changes at the defaults, a full system for production, and an empty one for destruction."""
    trajectory = _run(ccc_model(), 1000)
    changes = trajectory.values('update.changes')[1:]
    assert np.mean(changes == 0) > 0.7
    assert np.sum(changes > 5) > 5
    productive = _run(ccc_model(r_plus=30.0, r_minus=2.0), 200).values('update.diversity')[-1]
    destructive = _run(ccc_model(r_plus=2.0, r_minus=30.0), 200).values('update.diversity')[-1]
    assert productive > 90
    assert destructive < 25
