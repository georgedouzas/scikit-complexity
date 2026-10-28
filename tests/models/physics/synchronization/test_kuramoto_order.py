"""Test the `_kuramoto_order` module of `skcomplex.models.physics.synchronization`."""

import numpy as np

from skcomplex.analysis import AlignmentAnalysis
from skcomplex.models import run
from skcomplex.models.physics.synchronization import KuramotoOrderAnalysis


def test_kuramoto_order_equals_the_alignment():
    """Test that Kuramoto's order parameter of oscillators equals the alignment of kind `'mean-phase'`."""
    trajectory = run('kuramoto', settings={'until': 3}).trajectory_
    measure = KuramotoOrderAnalysis(['oscillators.theta']).check().prepare(trajectory).run()
    alignment = AlignmentAnalysis('mean-phase', ['oscillators.theta']).check().prepare(trajectory).run()
    np.testing.assert_array_equal(measure.order_, alignment.order_)
    assert measure.order_.shape == (len(trajectory.times),)
    np.testing.assert_array_equal(measure.phase_, alignment.phase_)
