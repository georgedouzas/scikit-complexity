"""Test the Lyapunov exponents: spectra of flows and maps, their invariants, and the largest exponent of data."""

import math
import time

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from skcomplex.core.analysis import lyapunov
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.errors import AnalysisError
from skcomplex.core.systems import System
from skcomplex.systems.mathematics.dynamical_systems import (
    HenonMapSystem,
    LogisticMapSystem,
    LorenzSystem,
    SprottJerkSystem,
)
from skcomplex.systems.physics.fluids import KuramotoSivashinskySystem
from skcomplex.systems.physics.nonlinear_oscillators import VanDerPolSystem


def _spectrum(source, **settings):
    """Return the Lyapunov exponents of a system."""
    return lyapunov(source, **settings).exponents


def _lorenz_states(rho: float, step: float, duration: float) -> np.ndarray:
    """Return states of the Lorenz-63 flow every `step` after a transient of 100, one row per sample."""

    def drift(_, state):
        x, y, z = state
        return [10 * (y - x), x * (rho - z) - y, x * y - 8 / 3 * z]

    times = np.arange(100, 100 + duration, step)
    solution = solve_ivp(drift, (0, times[-1]), [1, 1, 1], t_eval=times, rtol=1e-9, atol=1e-9)
    return solution.y.T


def _decay(size: int) -> System:
    """Return a group of members that each decay at the rate `x + x^3`."""
    return System(
        elements=[Element('group', values={'x': np.linspace(1.0, 2.0, size)}, size=size)],
        interactions=[
            Interaction(
                'decay',
                roles={'target': 'group'},
                laws=lambda s: [RateLaw(s.target.x.diff(s.t), -s.target.x - s.target.x**3)],
            ),
        ],
    )


def _logistic(r: float, size: int) -> np.ndarray:
    """Return iterates of the logistic map after a transient of 500."""
    values = np.empty(size + 500)
    values[0] = 0.3
    for t in range(1, values.size):
        values[t] = r * values[t - 1] * (1 - values[t - 1])
    return values[500:]


def test_lorenz_largest_exponent():
    """Test that the largest exponent of Lorenz-63 with the standard parameters is within 0.05 of 0.906."""
    exponents = _spectrum(LorenzSystem())
    assert exponents[0] == pytest.approx(0.906, abs=0.05)
    assert exponents[1] == pytest.approx(0.0, abs=0.02)
    assert exponents.sum() == pytest.approx(-(10 + 1 + 8 / 3), abs=0.05)


def test_logistic_map_at_four():
    """Test that the logistic map at r = 4 has exponent ln 2 within 0.01."""
    assert _spectrum(LogisticMapSystem(r=4.0), until=100_000)[0] == pytest.approx(np.log(2), abs=0.01)


def test_henon_map_spectrum():
    """Test that the Hénon map's exponents match the published 0.419 and -1.62."""
    exponents = _spectrum(HenonMapSystem(), until=100_000)
    assert exponents == pytest.approx([0.419, -1.623], abs=0.01)


def test_limit_cycle_is_not_chaotic():
    """Test that the Van der Pol limit cycle has a largest exponent of 0, not a positive one."""
    assert _spectrum(VanDerPolSystem().prepare().system_, n_exponents=1, until=500)[0] == pytest.approx(0.0, abs=0.01)


def test_sweep_of_a_map_reuses_its_compiled_tangent_dynamics():
    """Test that ten spectra of the logistic map after a first call take under a second and match fresh values."""
    _spectrum(LogisticMapSystem(r=3.9), until=2000)
    rates = np.linspace(3.5, 4.0, 10)
    start = time.perf_counter()
    exponents = [_spectrum(LogisticMapSystem(r=r), until=2000)[0] for r in rates]
    assert time.perf_counter() - start < 1.0
    assert exponents[-1] == pytest.approx(_spectrum(LogisticMapSystem(r=4.0), until=2000)[0])


def test_laws_with_absolute_values_have_a_spectrum():
    """Test that the Sprott jerk system, whose law has an absolute value, has three finite exponents of negative sum."""
    exponents = _spectrum(SprottJerkSystem(), until=200)
    assert exponents.shape == (3,)
    assert np.all(np.isfinite(exponents))
    assert exponents.sum() < 0


def test_kolmogorov_sinai_entropy_of_a_map_and_a_flow():
    """Test that the logistic map at r = 4 gives ln 2 and the Lorenz flow 0.906."""
    logistic = lyapunov(LogisticMapSystem(r=4.0, x=0.3), until=20_000, dt=1)
    flow = lyapunov(LorenzSystem(), until=500)
    assert logistic.kolmogorov_sinai_entropy == pytest.approx(math.log(2), abs=0.01)
    assert flow.kolmogorov_sinai_entropy == pytest.approx(0.906, abs=0.05)


def test_kaplan_yorke_dimension_of_a_map_and_a_flow():
    """Test that the Hénon map gives 1.26 and the Lorenz flow 2.06, and that exponents pass through as a spectrum."""
    henon = lyapunov(HenonMapSystem(), until=20_000, dt=1)
    flow = lyapunov(LorenzSystem(), until=500)
    assert henon.kaplan_yorke_dimension == pytest.approx(1.26, abs=0.02)
    assert flow.kaplan_yorke_dimension == pytest.approx(2.06, abs=0.03)
    assert lyapunov([-1.0, -2.0]).kaplan_yorke_dimension == 0
    assert lyapunov([0.5, 0.0]).kaplan_yorke_dimension == 2


@pytest.mark.slow
def test_largest_lyapunov_exponent_from_data():
    """Test the largest exponents of the logistic map at r = 4 and of the Lorenz x from recorded series."""
    logistic = lyapunov(_logistic(4.0, 10_000), kind='largest', dimension=2, trajectory_length=4, min_separation=10)
    assert logistic.largest_exponent[0] == pytest.approx(math.log(2), abs=0.1)
    series = _lorenz_states(28.0, 0.05, 300.0)[:6000, 0]
    flow = lyapunov(
        series,
        kind='largest',
        dt=0.05,
        dimension=5,
        delay=2,
        trajectory_length=60,
        min_separation=30,
        skip=10,
    )
    assert flow.largest_exponent[0] == pytest.approx(0.906, abs=0.15)


@pytest.mark.parametrize('size', [3, 100])
def test_spectrum_of_a_group_below_and_above_the_member_limit(size):
    """Test that a group of decaying members has exponents of -1, written out or from the backend Jacobian."""
    exponents = _spectrum(_decay(size), n_exponents=3, until=20)
    assert exponents == pytest.approx([-1.0, -1.0, -1.0], abs=0.01)


def test_spectrum_of_a_spectral_field_with_the_default_modes():
    """Test that Kuramoto Sivashinsky with its 128 modes gives finite exponents in decreasing order."""
    exponents = _spectrum(KuramotoSivashinskySystem(), n_exponents=2, until=1, transient=0)
    assert exponents.shape == (2,)
    assert np.all(np.isfinite(exponents))
    assert exponents[0] >= exponents[1]


@pytest.mark.slow
def test_largest_exponent_of_kuramoto_sivashinsky_at_length_22():
    """Test that Kuramoto Sivashinsky on a line of length 22 has a largest exponent near 0.04."""
    system = KuramotoSivashinskySystem(modes=24, length=22.0)
    exponents = _spectrum(system, n_exponents=1, until=200, dt=0.02, transient=50)
    assert exponents[0] == pytest.approx(0.04, abs=0.02)


def test_settings_the_kind_ignores_are_rejected():
    """Test that an embedding setting of the spectrum is rejected, with the settings it reads."""
    with pytest.raises(AnalysisError, match='does not read dimension'):
        lyapunov([0.5, 0.0], dimension=4)
