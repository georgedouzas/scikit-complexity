"""Test Binder cumulants, their crossings, sweeps of an order parameter, and replica overlaps."""

import math

import numpy as np
import pytest

from skcomplex.analysis import BinderAnalysis
from skcomplex.core.errors import AnalysisError
from skcomplex.engine.runtime import Simulation
from skcomplex.models import create

from .conftest import ISING_SIZES, ISING_TEMPERATURES

CRITICAL = 2 / math.log(1 + math.sqrt(2))


def _binder(kind: str, source: object, **settings: object) -> BinderAnalysis:
    """Return the run Binder analysis of a source for one kind."""
    return BinderAnalysis(kind, **settings).check().prepare(source).run()


def _kuramoto_order(rng: np.random.Generator, coupling: float, oscillators: int) -> np.ndarray:
    """Return the order parameter of Kuramoto oscillators with Lorentzian frequencies of width 1 after relaxation."""
    frequencies = np.tan(np.pi * (rng.random(oscillators) - 0.5))
    phases = rng.uniform(-np.pi, np.pi, oscillators)
    orders = []
    for step in range(6000):
        order = np.mean(np.exp(1j * phases))
        phases = phases + 0.05 * (frequencies + coupling * np.abs(order) * np.sin(np.angle(order) - phases))
        if step >= 2000:
            orders.append(np.abs(order))
    return np.asarray(orders)


def test_binder_crossings_ising_critical_point(ising_magnetizations):
    """Test that the Binder cumulants of Ising lattices of sides 16, 32, and 64 cross within 1 percent of T_c."""
    cumulants = np.column_stack(
        [_binder('cumulant', np.column_stack(ising_magnetizations[side])).cumulant_ for side in ISING_SIZES],
    )
    crossings = _binder('crossings', cumulants, parameters=ISING_TEMPERATURES).crossings_
    np.testing.assert_allclose(crossings, CRITICAL, rtol=0.01)
    with pytest.raises(AnalysisError, match='at least 2 sizes'):
        _binder('crossings', cumulants[:, :1], parameters=ISING_TEMPERATURES)


def test_binder_cumulant_of_an_ordered_phase():
    """Test that samples at ±1 give 2/3."""
    assert _binder('cumulant', [1.0, -1.0, 1.0]).cumulant_[0] == pytest.approx(2 / 3)


@pytest.mark.slow
def test_sweep_same_susceptibility_peak_on_two_transitions(ising_magnetizations):
    """Test that the susceptibility peaks of a 64 by 64 Ising lattice and of Kuramoto oscillators locate T_c and K_c."""
    ising = _binder(
        'sweep',
        np.column_stack(ising_magnetizations[64]),
        parameters=ISING_TEMPERATURES,
        size=64**2,
        temperatures=ISING_TEMPERATURES,
    )
    assert ising.peak_ == pytest.approx(CRITICAL, rel=0.03)
    couplings = np.linspace(1.0, 3.0, 11)
    rng = np.random.default_rng(4)
    orders = np.column_stack([_kuramoto_order(rng, coupling, 2000) for coupling in couplings])
    kuramoto = _binder('sweep', orders, parameters=couplings, size=2000)
    assert kuramoto.peak_ == pytest.approx(2, rel=0.1)
    assert np.all(np.diff(kuramoto.order_[couplings >= 2.2]) > 0)


def test_replica_overlap_of_ising_lattices():
    """Test that two replicas of a 32 by 32 Ising lattice overlap at temperature 1.5 and not at temperature 5."""
    overlaps = {}
    for temperature in (1.5, 5.0):
        replicas = [
            Simulation(create('ising', size=32, temperature=temperature, initial='up'), dt=1, random_state=seed)
            .run(until=500)
            .trajectory_.lattice('spins.spin')[100:]
            for seed in (0, 1)
        ]
        overlaps[temperature] = float(np.mean(np.abs(_binder('overlap', np.stack(replicas)).overlap_)))
    assert overlaps[1.5] > 0.9
    assert overlaps[5.0] < 0.1
    assert _binder('overlap', [[1, -1], [1, -1]]).overlap_ == 1.0
    with pytest.raises(AnalysisError, match='2 replicas'):
        _binder('overlap', [[1, -1], [1, -1], [1, 1]])


def test_invalid_settings_are_rejected():
    """Test an unknown kind, an ignored setting, and a sweep without control parameter values."""
    with pytest.raises(AnalysisError, match='cumulant, crossings'):
        BinderAnalysis('susceptibility').check()
    with pytest.raises(AnalysisError, match='does not read size'):
        BinderAnalysis('cumulant', size=100).check()
    with pytest.raises(AnalysisError, match='parameters'):
        BinderAnalysis('sweep').check()
