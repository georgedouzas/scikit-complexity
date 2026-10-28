"""Test the collapse analysis of finite-size scaling, aging, distributions scaled by age, and crossings."""

import math

import numpy as np
import pytest

from skcomplex.analysis import ClusterAnalysis, CollapseAnalysis, CorrelationAnalysis, DistributionAnalysis
from skcomplex.core.errors import AnalysisError
from skcomplex.engine.runtime import Simulation
from skcomplex.models import create

from .conftest import ISING_SIZES, ISING_TEMPERATURES

CRITICAL = 2 / math.log(1 + math.sqrt(2))
PERCOLATION = 0.5927


def _collapse(kind: str, source: object, **settings: object) -> CollapseAnalysis:
    """Return the collapse analysis of a source for one kind, run."""
    return CollapseAnalysis(kind, **settings).check().prepare(source).run()


def _susceptibilities(magnetizations: dict[int, list[np.ndarray]]) -> dict[int, np.ndarray]:
    """Return the susceptibility `N var(|m|) / T` of every size at every temperature."""
    moments = DistributionAnalysis('moments', absolute=True).check()
    return {
        side: side**2 * moments.prepare(np.column_stack(samples)).run().variance_ / np.asarray(ISING_TEMPERATURES)
        for side, samples in magnetizations.items()
    }


def _spanning(grids: list[np.ndarray]) -> float:
    """Return the share of grids with a cluster joining the opposite edges of their first axis."""
    return ClusterAnalysis('spanning').check().prepare(grids).run().probability_


def _trap_renewals(rng: np.random.Generator, walkers: int, horizon: float) -> list[np.ndarray]:
    """Return the jump times of walkers in Bouchaud's trap model with T / T_g = 0.5, waiting times of exponent 1/2."""
    jumps = []
    for _ in range(walkers):
        times = np.cumsum(rng.random(64) ** (-1 / 0.5))
        while times[-1] < horizon:
            times = np.concatenate([times, times[-1] + np.cumsum(rng.random(64) ** (-1 / 0.5))])
        jumps.append(times)
    return jumps


def test_scaling_collapse_same_on_two_transitions(ising_magnetizations):
    """Test that the Ising susceptibility and the spanning probability collapse better than with wrong exponents."""
    susceptibilities = _susceptibilities(ising_magnetizations)
    grid = ISING_TEMPERATURES
    exact = _collapse('scaling', susceptibilities, parameters=grid, critical=CRITICAL, nu=1.0, exponent=7 / 4)
    wrong = _collapse('scaling', susceptibilities, parameters=grid, critical=CRITICAL, nu=1.2, exponent=7 / 4 * 1.2)
    fitted = _collapse(
        'scaling',
        susceptibilities,
        parameters=grid,
        critical=CRITICAL,
        nu=1.3,
        exponent=7 / 4,
        fit=('nu',),
    )
    assert exact.quality_ < wrong.quality_
    assert fitted.nu_ == pytest.approx(1, rel=0.1)
    rng = np.random.default_rng(0)
    probabilities = np.linspace(0.56, 0.62, 13)
    spanning = {
        side: np.asarray(
            [_spanning([rng.random((side, side)) < p for _ in range(600)]) for p in probabilities],
        )
        for side in (16, 32, 64)
    }
    exact = _collapse('scaling', spanning, parameters=probabilities, critical=PERCOLATION, nu=4 / 3)
    wrong = _collapse('scaling', spanning, parameters=probabilities, critical=PERCOLATION, nu=4 / 3 * 1.2)
    fitted = _collapse('scaling', spanning, parameters=probabilities, critical=PERCOLATION, nu=1.0, fit=('nu',))
    assert exact.quality_ < wrong.quality_
    assert fitted.nu_ == pytest.approx(4 / 3, rel=0.1)
    with pytest.raises(AnalysisError, match='at least 2 sizes'):
        _collapse('scaling', {32: spanning[32]}, parameters=probabilities, critical=PERCOLATION, nu=1.0)


@pytest.mark.slow
def test_two_time_correlation_aging_on_two_domains():
    """Test that an Ising quench and the trap model age: correlations grow with the age and collapse on tau / t_w."""
    system = create('ising', size=256, temperature=1.0, sampler='glauber')
    # One quench gives a ratio spread between 0.01 and 0.13 by seed, so the curves average four seeded quenches.
    quenches = [Simulation(system, dt=1, random_state=seed).check().run(until=400) for seed in range(4, 8)]
    runs = [quench.trajectory_.lattice('spins.spin') for quench in quenches]
    waiting, lags = [50, 100, 200], np.arange(0, 200)
    ising = CorrelationAnalysis('two-time', waiting=waiting, lags=lags).check().prepare(np.stack(runs)).run()
    horizon = 1000.0
    renewals = _trap_renewals(np.random.default_rng(1), 20_000, horizon)
    grid = np.arange(0, 1001, 5.0)
    counts = np.asarray([np.searchsorted(times, grid, side='right') for times in renewals])
    trap_waiting = [20, 40, 80]
    trap_lags = np.arange(0, 21)
    trap_curves = np.asarray(
        [[np.mean(counts[:, start] == counts[:, start + lag]) for lag in trap_lags] for start in trap_waiting],
    )
    cases = ((ising.correlation_, waiting, lags, 1.0), (trap_curves, trap_waiting, trap_lags, 5.0))
    for curves, starts, steps, scale in cases:
        assert np.all(np.diff(curves[:, 10]) > 0)
        collapse = _collapse('aging', curves, waiting=np.asarray(starts) * scale, lags=steps * scale)
        assert collapse.ratio_spread_ < 0.05
        assert collapse.time_spread_ > collapse.ratio_spread_


def test_collapse_distributions_trap_model():
    """Test that the trapping times of the traps occupied at ages 10^3, 10^4, and 10^5 collapse within 10 percent."""
    renewals = _trap_renewals(np.random.default_rng(2), 20_000, 1e5)
    samples = {}
    for age in (1e3, 1e4, 1e5):
        durations = []
        for times in renewals:
            index = np.searchsorted(times, age)
            start = times[index - 1] if index > 0 else 0.0
            durations.append(times[index] - start)
        samples[age] = np.asarray(durations)
    collapse = _collapse('distribution', samples, ratios=np.geomspace(0.3, 3, 5))
    assert collapse.spread_ < 0.1
    assert collapse.densities_.shape == (3, 4)


def test_crossings_of_moment_ratios_at_the_ising_critical_point(ising_magnetizations):
    """Test that the moment-ratio cumulants of Ising lattices of sides 16, 32, and 64 cross within 1 percent of T_c."""
    ratio = DistributionAnalysis('moment-ratio').check()
    cumulants = {
        side: ratio.prepare(np.column_stack(ising_magnetizations[side])).run().cumulant_ for side in ISING_SIZES
    }
    crossings = _collapse('crossings', cumulants, parameters=ISING_TEMPERATURES).crossings_
    np.testing.assert_allclose(crossings, CRITICAL, rtol=0.01)
    np.testing.assert_array_equal(
        _collapse('crossings', [cumulants[side] for side in ISING_SIZES], parameters=ISING_TEMPERATURES).crossings_,
        crossings,
    )
    with pytest.raises(AnalysisError, match='at least 2 sizes'):
        _collapse('crossings', {16: cumulants[16]}, parameters=ISING_TEMPERATURES)


def test_unknown_kind_ignored_setting_and_missing_settings():
    """Test that an unknown kind, an ignored setting, and missing settings raise errors."""
    with pytest.raises(AnalysisError, match='scaling, aging, distribution'):
        CollapseAnalysis('unknown').check()
    with pytest.raises(AnalysisError, match='does not read ratios'):
        CollapseAnalysis('aging', waiting=[1], lags=[1], ratios=[0.1, 1.0]).check()
    with pytest.raises(AnalysisError, match='needs parameters, critical, and nu'):
        CollapseAnalysis('scaling').check()
    with pytest.raises(AnalysisError, match='crossings needs the control parameter values'):
        CollapseAnalysis('crossings').check()
    with pytest.raises(AnalysisError, match="fit='gamma'"):
        CollapseAnalysis(parameters=[0.0], critical=0.0, nu=1.0, fit=('gamma',)).check()
