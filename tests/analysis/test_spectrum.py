"""Test the spectrum analysis of series, fields, and field models."""

import numpy as np
import pytest

from skcomplex.analysis import SpectrumAnalysis
from skcomplex.core.errors import AnalysisError
from skcomplex.models import run


def _spectrum(kind: str, source: object, **settings: object) -> SpectrumAnalysis:
    """Return the spectrum analysis of a source for one kind, run."""
    return SpectrumAnalysis(kind, **settings).check().prepare(source).run()


def test_power_spectrum_of_a_sine():
    """Test that a sine has its frequency's peak."""
    times = np.arange(8192) * 0.01
    spectrum = _spectrum('power', np.sin(2 * np.pi * 3 * times), dt=0.01)
    assert spectrum.frequencies_[spectrum.power_[0].argmax()] == pytest.approx(3, abs=0.1)


def test_structure_factor_of_stripes():
    """Test that stripes of period 16 give their peak."""
    stripes = np.sign(np.cos(2 * np.pi * np.arange(128) / 16))[:, None] * np.ones((128, 128))
    factor = _spectrum('structure-factor', stripes)
    assert 2 * np.pi / factor.wavenumbers_[factor.power_.argmax()] == pytest.approx(16)


def test_structure_factor_same_on_a_trajectory_and_its_last_frame():
    """Test that an Ising trajectory and the array of its last frame give the same structure factor."""
    spins = run('ising', size=32, settings={'until': 5}).trajectory_
    np.testing.assert_array_equal(
        _spectrum('structure-factor', spins).power_,
        _spectrum('structure-factor', spins.lattice('spins.spin')[-1]).power_,
    )


def test_unknown_kind_and_ignored_setting():
    """Test that an unknown kind and a setting the kind ignores raise errors."""
    with pytest.raises(AnalysisError, match='power, structure-factor'):
        SpectrumAnalysis('unknown').check()
    with pytest.raises(AnalysisError, match='does not read spacing'):
        SpectrumAnalysis('power', spacing=2.0).check()
