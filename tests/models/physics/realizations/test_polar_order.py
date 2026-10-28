"""Test the `_polar_order` module of `skcomplex.models.physics.realizations`."""

import numpy as np

from skcomplex.analysis import AlignmentAnalysis
from skcomplex.models import run
from skcomplex.models.physics.realizations import PolarOrderAnalysis


def test_polar_order_equals_the_alignment():
    """Test that the polar order of a Vicsek flock equals the alignment of kind `'mean-vector'`."""
    trajectory = run('vicsek', settings={'until': 3}).trajectory_
    measure = PolarOrderAnalysis().check().prepare(trajectory).run()
    alignment = AlignmentAnalysis('mean-vector').check().prepare(trajectory).run()
    np.testing.assert_array_equal(measure.order_, alignment.order_)
    assert measure.order_.shape == (len(trajectory.times),)
    np.testing.assert_allclose(measure.order_, trajectory.values('polarization'), atol=1e-9)
