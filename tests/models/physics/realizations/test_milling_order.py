"""Test the `_milling_order` module of `skcomplex.models.physics.realizations`."""

import numpy as np

from skcomplex.analysis import AlignmentAnalysis
from skcomplex.models import run
from skcomplex.models.physics.realizations import MillingOrderAnalysis


def test_milling_order_equals_the_alignment():
    """Test that the milling order of a flock of boids equals the alignment of kind `'mean-rotation'`."""
    trajectory = run('boids', settings={'until': 3}).trajectory_
    measure = MillingOrderAnalysis().check().prepare(trajectory).run()
    alignment = AlignmentAnalysis('mean-rotation').check().prepare(trajectory).run()
    np.testing.assert_array_equal(measure.order_, alignment.order_)
    assert measure.order_.shape == (len(trajectory.times),)
