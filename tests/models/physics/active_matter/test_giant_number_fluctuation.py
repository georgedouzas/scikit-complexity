"""Test the `_giant_number_fluctuation` module of `skcomplex.models.physics.active_matter`."""

import numpy as np
import pytest

from skcomplex.analysis import ScalingAnalysis
from skcomplex.models.physics.active_matter import GiantNumberFluctuationAnalysis


def test_giant_number_fluctuation_equals_the_variance_mean_law():
    """Test that the number fluctuations equal the variance-mean law of box counts, 1/2 for Poisson counts."""
    counts = np.random.default_rng(0).poisson(5.0, (20, 64, 64))
    fluctuation = GiantNumberFluctuationAnalysis(scales=[1, 2, 4, 8]).check().prepare(counts).run()
    general = ScalingAnalysis('variance-mean', scales=[1, 2, 4, 8]).check().prepare(counts).run()
    for name in ('scales_', 'means_', 'deviations_', 'exponent_'):
        np.testing.assert_array_equal(getattr(fluctuation, name), getattr(general, name))
    assert fluctuation.exponent_ == pytest.approx(0.5, abs=0.05)
