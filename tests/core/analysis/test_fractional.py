"""Test the Mittag-Leffler function and the exact solution of linear fractional relaxation."""

import math

import numpy as np
import pytest

from skcomplex.core.analysis import equations_of, mittag_leffler, solve_symbolically
from skcomplex.core.base import AnalysisError, parameter, state
from skcomplex.core.components import Element
from skcomplex.core.laws import Rate
from skcomplex.core.system import System


class Relaxing(Element):
    """Element whose state relaxes with a Caputo derivative of order 0.6."""

    x: float = state(2.0, order=0.6)
    lam: float = parameter(0.5)

    def laws(self, s):
        """Return the relaxation rate."""
        return [Rate(s.x, -s.lam * s.x)]


class Saturating(Element):
    """Element whose fractional state follows a nonlinear law."""

    x: float = state(0.5, order=0.6)

    def laws(self, s):
        """Return the logistic rate."""
        return [Rate(s.x, s.x * (1 - s.x))]


def test_mittag_leffler_exponential_case():
    """Test that E_(1, 1)(z) equals exp(z) within 1e-12."""
    z = np.linspace(-20.0, 5.0, 51)
    assert np.max(np.abs(mittag_leffler(1.0, 1.0, z) - np.exp(z))) < 1e-12


@pytest.mark.parametrize(
    ('alpha', 'beta', 'z', 'expected'),
    [
        (0.5, 1.0, -1.0, math.exp(1.0) * math.erfc(1.0)),
        (0.5, 1.0, -10.0, math.exp(100.0) * math.erfc(10.0)),
        (2.0, 1.0, -4.0, math.cos(2.0)),
        (2.0, 2.0, -4.0, math.sin(2.0) / 2.0),
        (1.0, 2.0, 1.5, math.expm1(1.5) / 1.5),
    ],
)
def test_mittag_leffler_closed_forms(alpha, beta, z, expected):
    """Test that the function matches its closed forms in error functions and trigonometric functions."""
    assert float(mittag_leffler(alpha, beta, z)) == pytest.approx(expected, rel=1e-12)


def test_mittag_leffler_asymptotic_matches_series():
    """Test that the asymptotic expansion for large arguments continues the series without a jump."""
    alpha = 0.6
    edge = -(50.0**alpha)
    below, above = mittag_leffler(alpha, 1.0, np.array([edge * 1.0001, edge * 0.9999]))
    assert below == pytest.approx(above, rel=1e-3)
    assert float(mittag_leffler(alpha, 1.0, -1e6)) == pytest.approx(1e-6 / math.gamma(1 - alpha), rel=1e-5)


def test_mittag_leffler_complex_arguments():
    """Test that complex arguments give complex values, with E_(2, 1)(-z^2) = cos(z)."""
    value = mittag_leffler(2.0, 1.0, np.array([-((1.0 + 0.5j) ** 2)]))
    assert np.iscomplexobj(value)
    assert value[0] == pytest.approx(np.cos(1.0 + 0.5j), rel=1e-12)


def test_mittag_leffler_rejects_nonpositive_alpha():
    """Test that a nonpositive first parameter is rejected."""
    with pytest.raises(AnalysisError, match='alpha'):
        mittag_leffler(0.0, 1.0, 1.0)


def test_solve_symbolically_fractional_relaxation():
    """Test that linear fractional relaxation is solved by the Mittag-Leffler function of its order."""
    system = System(components=[Relaxing('r')])
    solution = solve_symbolically(system, use_initial_state=True)['r.x']
    assert str(solution) == 'x_r(t) == 2*mittag_leffler(3/5, 1, -lam_r*t^(3/5))'
    symbols = {str(item): item for item in solution.rhs().variables()}
    value = float(solution.rhs().subs({symbols['t']: 2.0, symbols['lam_r']: 0.5}))
    assert value == pytest.approx(2 * float(mittag_leffler(0.6, 1.0, -0.5 * 2.0**0.6)), rel=1e-12)
    assert 'x_r(0)' in str(solve_symbolically(system)['r.x'])


def test_equations_of_fractional_state_write_caputo_derivative():
    """Test that the equation of a fractional state writes its Caputo derivative of the exact order."""
    assert str(equations_of(System(components=[Relaxing('r')]))['r.x']) == 'caputo(x_r(t), t, 3/5) == -lam_r*x_r(t)'


def test_solve_symbolically_rejects_nonlinear_fractional_law():
    """Test that a nonlinear fractional law has no exact solution."""
    with pytest.raises(AnalysisError, match='linear fractional relaxation'):
        solve_symbolically(System(components=[Saturating('s')]))
