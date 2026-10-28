"""Test the `_nematic_order` module of `skcomplex.models.physics.realizations`."""

import numpy as np

from skcomplex.analysis import AlignmentAnalysis
from skcomplex.models import run
from skcomplex.models.physics.realizations import NematicOrderAnalysis


def test_nematic_order_equals_the_alignment():
    """Test that the nematic order of active nematics equals the alignment of kind `'axial'`."""
    trajectory = run('ginelli_nematics', settings={'until': 3}).trajectory_
    measure = NematicOrderAnalysis().check().prepare(trajectory).run()
    alignment = AlignmentAnalysis('axial').check().prepare(trajectory).run()
    np.testing.assert_array_equal(measure.order_, alignment.order_)
    assert measure.order_.shape == (len(trajectory.times),)
