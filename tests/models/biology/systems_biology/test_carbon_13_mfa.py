"""Test the `_carbon_13_mfa` module of `skcomplex.models.biology.systems_biology`."""

import numpy as np
import pytest

from skcomplex.models.biology.systems_biology import carbon_13_mfa, tca_labelling_analysis

TRUE = {'v1': 100.0, 'v2': 100.0, 'v3': 50.0, 'v4': 50.0, 'v5': 50.0, 'v6': 50.0, 'v8': 50.0, 'v9': 50.0}


def test_estimation_recovers_the_split_of_alpha_ketoglutarate():
    """Test the estimated split, that every interval holds its true value, and that the test accepts the fit."""
    analysis = tca_labelling_analysis(noise=0.002, random_state=0).check().prepare(carbon_13_mfa()).run()
    assert analysis.fluxes_['v3'] == pytest.approx(50, abs=2)
    assert analysis.fluxes_['v4'] == pytest.approx(50, abs=2)
    assert analysis.accepted_
    for name, value in TRUE.items():
        low, high = analysis.intervals_[name]
        assert low - 1e-6 <= value <= high + 1e-6, name


def test_simulation_gives_the_published_pattern():
    """Test the glutamate of the true fluxes and the tracer of Antoniewicz et al. (2007)."""
    analysis = tca_labelling_analysis('simulation').check().prepare(carbon_13_mfa()).run()
    found = analysis.distributions_['published']['Glu']
    np.testing.assert_allclose(found, [0.3464, 0.2695, 0.2708, 0.0807, 0.0286, 0.0039], atol=5e-5)
