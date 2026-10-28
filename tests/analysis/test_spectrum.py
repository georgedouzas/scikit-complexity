"""Test the spectrum analysis of series, fields, and field models."""

import numpy as np
import pytest

from skcomplex.analysis import SpectrumAnalysis, fields_of
from skcomplex.core.elements import FieldElement
from skcomplex.core.errors import AnalysisError
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.system import System
from skcomplex.models import run
from skcomplex.models.chemistry.pattern_formation import BrusselatorElement


def _spectrum(kind: str, source: object, **settings: object) -> SpectrumAnalysis:
    """Return the spectrum analysis of a source for one kind, run."""
    return SpectrumAnalysis(kind, **settings).check().prepare(source).run()


def _brusselator(b, *, discretization='spectral', boundary='periodic'):
    """Return a one-dimensional Brusselator field at its uniform state `(1, b)`."""
    values = {'u': 1.0, 'v': b, 'a': 1.0, 'b': b, 'Du': 1.0, 'Dv': 8.0}
    space = LatticeSpace((128,), spacing=0.5, boundary=boundary)
    cells = FieldElement('cells', element=BrusselatorElement, space=space, values=values, discretization=discretization)
    return System(elements=[cells])


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
        _spectrum('structure-factor', fields_of(spins)[-1]).power_,
    )


def test_dispersion_relation_finds_the_turing_band():
    """Test that the Brusselator above its Turing threshold grows on the analytic band within one grid wavenumber."""
    relation = _spectrum('dispersion', _brusselator(1.9))
    gap = np.sqrt(6.2**2 - 32)
    low, high = np.sqrt((6.2 - gap) / 16), np.sqrt((6.2 + gap) / 16)
    step = 2 * np.pi / (128 * 0.5)
    assert relation.uniform_ == pytest.approx({'u': 1.0, 'v': 1.9})
    assert relation.band_.min() == pytest.approx(low, abs=step)
    assert relation.band_.max() == pytest.approx(high, abs=step)
    assert relation.growth_rates_[relation.magnitudes_ == 0] < 0


def test_dispersion_relation_of_finite_differences_and_below_threshold():
    """Test that finite differences agree with spectral modes at long waves, and that no mode grows below threshold."""
    spectral = _spectrum('dispersion', _brusselator(1.9))
    stencil = _spectrum('dispersion', _brusselator(1.9, discretization='finite-difference'))
    long = spectral.magnitudes_ < 0.3
    assert stencil.growth_rates_[long] == pytest.approx(spectral.growth_rates_[long], abs=1e-2)
    assert _spectrum('dispersion', _brusselator(1.7)).band_.size == 0


def test_dispersion_relation_rejects_non_periodic_fields_and_data():
    """Test that a field with walls has no Fourier modes, and that an array is not a system."""
    with pytest.raises(AnalysisError, match='periodic lattice'):
        _spectrum('dispersion', _brusselator(1.9, boundary='fixed'))
    with pytest.raises(AnalysisError, match='from a system'):
        _spectrum('dispersion', np.zeros(4))


def test_unknown_kind_and_ignored_setting():
    """Test that an unknown kind and a setting the kind ignores raise errors."""
    with pytest.raises(AnalysisError, match='power, structure-factor, dispersion'):
        SpectrumAnalysis('unknown').check()
    with pytest.raises(AnalysisError, match='does not read spacing'):
        SpectrumAnalysis('power', spacing=2.0).check()
