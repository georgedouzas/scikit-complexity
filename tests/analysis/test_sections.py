"""Test the section analysis: Poincaré sections, stroboscopic samples, and orbit diagrams."""

import math

import numpy as np
import pytest

from skcomplex.analysis import SectionAnalysis
from skcomplex.core.errors import AnalysisError
from skcomplex.models.biology.ecology import logistic_growth
from skcomplex.models.mathematics.dynamical_systems import logistic_map
from skcomplex.models.physics.mechanics import harmonic_oscillator
from skcomplex.models.physics.nonlinear_oscillators import duffing


def _section(kind, source, **settings):
    """Return the section analysis of a source for one kind, run."""
    return SectionAnalysis(kind, **settings).check().prepare(source).run()


def _orbit_diagram(source, parameter, values, observable, **settings):
    """Return the orbit diagram of a source over the values of a parameter."""
    return _section(
        'orbit-diagram',
        source,
        parameter=parameter,
        parameter_values=values,
        observable=observable,
        **settings,
    )


@pytest.mark.slow
def test_forced_duffing_stroboscopic_section_is_fractal():
    """Test that the chaotic forced Duffing section has thousands of distinct points, not a periodic orbit."""
    section = _section('stroboscopic', duffing(), period=2 * math.pi / 1.2, n_points=2000)
    points = np.round(np.column_stack([section.values_['duffing.x'], section.values_['duffing.x_d1']]), 4)
    assert len(section.times_) == 2000
    assert len(np.unique(points, axis=0)) > 1900


def test_periodic_duffing_section_is_finite():
    """Test that a weakly forced Duffing oscillator settles on a periodic orbit with a single section point."""
    section = _section('stroboscopic', duffing(gamma=0.1), period=2 * math.pi / 1.2, n_points=50, transient=200)
    points = np.column_stack([section.values_['duffing.x'], section.values_['duffing.x_d1']])
    assert np.ptp(points, axis=0) == pytest.approx([0.0, 0.0], abs=1e-4)


def test_harmonic_oscillator_section_crosses_upward():
    """Test that upward crossings of x = 0 by the harmonic oscillator come once per period with the full speed."""
    section = _section('poincare', harmonic_oscillator(), state='mass.x', direction=1, until=20.0)
    assert np.diff(section.times_) == pytest.approx(2 * math.pi, abs=1e-5)
    assert section.values_['mass.v'] == pytest.approx(np.ones(len(section.times_)), abs=1e-5)
    both = _section('poincare', harmonic_oscillator(), state='mass.x', direction=0, until=20.0)
    assert len(both.times_) == 2 * len(section.times_)


def test_section_discards_transient():
    """Test that crossings before the transient are discarded."""
    section = _section('poincare', harmonic_oscillator(), state='mass.x', until=20.0, transient=10.0)
    assert section.times_.min() >= 10.0


def test_section_reads_the_named_variables():
    """Test that a section reads the values of the named states only."""
    section = _section('poincare', harmonic_oscillator(), variables=['mass.v'], state='mass.x', until=20.0)
    assert list(section.values_) == ['mass.v']


def test_section_rejects_maps_and_unknown_states():
    """Test that maps and names that are not scalar states are rejected."""
    with pytest.raises(AnalysisError, match='continuous time'):
        _section('stroboscopic', logistic_map(), period=1.0)
    with pytest.raises(AnalysisError, match='not a scalar state'):
        _section('poincare', logistic_growth(), state='population.r')


def test_section_needs_its_settings():
    """Test that a stroboscopic section without a period, and a setting another kind reads, are rejected at check."""
    with pytest.raises(AnalysisError, match='needs period'):
        SectionAnalysis('stroboscopic').check()
    with pytest.raises(AnalysisError, match='does not read keep'):
        SectionAnalysis('poincare', state='mass.x', keep=3).check()


@pytest.mark.slow
def test_period_doubling_of_logistic_map():
    """Test that the logistic map has period 1 below 3, 2 between 3 and 3.449, and 4 between 3.449 and 3.544."""
    rates = np.linspace(2.8, 4.0, 241)
    diagram = _orbit_diagram(logistic_map, 'map.r', rates, 'map.x', transient=3000, keep=64, dt=1)
    periods = diagram.periods_
    assert len(diagram.points_) == 241
    assert np.all(periods[(rates > 2.8) & (rates < 2.98)] == 1)
    assert np.all(periods[(rates > 3.02) & (rates < 3.43)] == 2)
    assert np.all(periods[(rates > 3.47) & (rates < 3.53)] == 4)
    assert periods[rates > 3.9].min() > 32


def test_vectorised_orbit_diagram_matches_the_serial_sweep():
    """Test that the orbit diagram of the logistic map run as one ensemble equals the sweep of separate runs."""
    rates = np.linspace(2.8, 4.0, 200)
    serial = _orbit_diagram(logistic_map, 'map.r', rates, 'map.x', transient=200, keep=16, dt=1)
    vectorised = _orbit_diagram(logistic_map, 'map.r', rates, 'map.x', transient=200, keep=16, dt=1, vectorized=True)
    assert len(vectorised.points_) == 200
    for first, second in zip(serial.points_, vectorised.points_, strict=True):
        np.testing.assert_allclose(np.ravel(first), np.ravel(second), atol=1e-12)
