"""Test Poincaré sections and stroboscopic samples."""

import math

import numpy as np
import pytest

from skcomplex.core.analysis import poincare_section, stroboscopic
from skcomplex.core.base import AnalysisError
from skcomplex.models.ecology import logistic_growth
from skcomplex.models.nonlinear import duffing, logistic_map
from skcomplex.models.physics import harmonic_oscillator


def test_forced_duffing_stroboscopic_section_is_fractal():
    """Test that the chaotic forced Duffing section has thousands of distinct points, not a periodic orbit."""
    section = stroboscopic(duffing(), 2 * math.pi / 1.2, n_points=2000)
    points = np.round(np.column_stack([section['duffing.x'], section['duffing.x_d1']]), 4)
    assert len(section) == 2000
    assert len(np.unique(points, axis=0)) > 1900


def test_periodic_duffing_section_is_finite():
    """Test that a weakly forced Duffing oscillator settles on a periodic orbit with a single section point."""
    section = stroboscopic(duffing(gamma=0.1), 2 * math.pi / 1.2, n_points=50, transient=200)
    points = np.column_stack([section['duffing.x'], section['duffing.x_d1']])
    assert np.ptp(points, axis=0) == pytest.approx([0.0, 0.0], abs=1e-4)


def test_harmonic_oscillator_section_crosses_upward():
    """Test that upward crossings of x = 0 by the harmonic oscillator come once per period with the full speed."""
    section = poincare_section(harmonic_oscillator(), 'mass.x', direction=1, until=20.0)
    assert np.diff(section.times) == pytest.approx(2 * math.pi, abs=1e-6)
    assert section['mass.v'] == pytest.approx(np.ones(len(section)), abs=1e-5)
    both = poincare_section(harmonic_oscillator(), 'mass.x', direction=0, until=20.0)
    assert len(both) == 2 * len(section)


def test_section_discards_transient():
    """Test that crossings before the transient are discarded."""
    section = poincare_section(harmonic_oscillator(), 'mass.x', until=20.0, transient=10.0)
    assert section.times.min() >= 10.0


def test_section_rejects_maps_and_unknown_states():
    """Test that maps and names that are not scalar states are rejected."""
    with pytest.raises(AnalysisError, match='continuous time'):
        stroboscopic(logistic_map(), 1.0)
    with pytest.raises(AnalysisError, match='not a scalar state'):
        poincare_section(logistic_growth(), 'population.r')
