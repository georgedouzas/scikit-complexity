"""Test the section analysis: Poincaré sections and stroboscopic samples."""

import math

import numpy as np
import pytest

from skcomplex.core.analysis import section
from skcomplex.core.errors import AnalysisError
from skcomplex.systems.biology.ecology import logistic_growth
from skcomplex.systems.mathematics.dynamical_systems import LogisticMapSystem
from skcomplex.systems.physics.mechanics import harmonic_oscillator
from skcomplex.systems.physics.nonlinear_oscillators import duffing


def _section(kind, source, **settings):
    """Return the section of a source for one kind."""
    return section(source, kind=kind, **settings)


@pytest.mark.slow
def test_forced_duffing_stroboscopic_section_is_fractal():
    """Test that the chaotic forced Duffing section has thousands of distinct points, not a periodic orbit."""
    section = _section('stroboscopic', duffing(), period=2 * math.pi / 1.2, n_points=2000)
    points = np.round(np.column_stack([section.values['duffing.x'], section.values['duffing.x_d1']]), 4)
    assert len(section.times) == 2000
    assert len(np.unique(points, axis=0)) > 1900


def test_periodic_duffing_section_is_finite():
    """Test that a weakly forced Duffing oscillator settles on a periodic orbit with a single section point."""
    section = _section('stroboscopic', duffing(gamma=0.1), period=2 * math.pi / 1.2, n_points=50, transient=200)
    points = np.column_stack([section.values['duffing.x'], section.values['duffing.x_d1']])
    assert np.ptp(points, axis=0) == pytest.approx([0.0, 0.0], abs=1e-4)


def test_harmonic_oscillator_section_crosses_upward():
    """Test that upward crossings of x = 0 by the harmonic oscillator come once per period with the full speed."""
    section = _section('poincare', harmonic_oscillator(), state='mass.x', direction=1, until=20.0)
    assert np.diff(section.times) == pytest.approx(2 * math.pi, abs=1e-5)
    assert section.values['mass.x_d1'] == pytest.approx(np.ones(len(section.times)), abs=1e-5)
    both = _section('poincare', harmonic_oscillator(), state='mass.x', direction=0, until=20.0)
    assert len(both.times) == 2 * len(section.times)


def test_section_discards_transient():
    """Test that crossings before the transient are discarded."""
    section = _section('poincare', harmonic_oscillator(), state='mass.x', until=20.0, transient=10.0)
    assert section.times.min() >= 10.0


def test_section_reads_the_named_variables():
    """Test that a section reads the values of the named states only."""
    section = _section('poincare', harmonic_oscillator(), variables=['mass.x_d1'], state='mass.x', until=20.0)
    assert list(section.values) == ['mass.x_d1']


def test_section_rejects_maps_and_unknown_states():
    """Test that maps and names that are not scalar states are rejected."""
    with pytest.raises(AnalysisError, match='continuous time'):
        _section('stroboscopic', LogisticMapSystem(), period=1.0)
    with pytest.raises(AnalysisError, match='not a scalar state'):
        _section('poincare', logistic_growth(), state='population.r')


def test_section_needs_its_settings():
    """Test that a stroboscopic section without a period, and a setting another kind reads, are rejected."""
    with pytest.raises(AnalysisError, match='needs period'):
        section(harmonic_oscillator(), kind='stroboscopic')
    with pytest.raises(AnalysisError, match='does not read period'):
        section(harmonic_oscillator(), kind='poincare', state='mass.x', period=3.0)
