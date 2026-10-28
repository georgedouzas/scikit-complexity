"""Test the `_binder_cumulant` module of `skcomplex.models.physics.realizations`."""

import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.analysis import DistributionAnalysis
from skcomplex.models.physics.realizations import BinderCumulantAnalysis


def test_binder_cumulant_equals_the_moment_ratio():
    """Test that the Binder cumulant sets the attributes of the moment ratio of orders 4 and 2 with the same values."""
    samples = np.random.default_rng(0).standard_normal((500, 3)) * 0.3 + 0.2
    binder = BinderCumulantAnalysis().check().prepare(samples).run()
    general = DistributionAnalysis('moment-ratio', orders=(4, 2)).check().prepare(samples).run()
    np.testing.assert_array_equal(binder.cumulant_, general.cumulant_)
    np.testing.assert_array_equal(binder.ratio_, general.ratio_)
    assert BinderCumulantAnalysis().check().prepare([1.0, -1.0]).run().cumulant_[0] == pytest.approx(2 / 3)
    assert clone(binder).get_params() == {'group': None, 'variables': None}
