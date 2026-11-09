"""Test the large deviation analysis of generating functions, rate functions, cloning, and reweighting."""

import math
from typing import Annotated

import numpy as np
import pytest
from scipy import special

from skcomplex.core.analysis import LargeDeviationsResult, large_deviations
from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import JumpLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.errors import AnalysisError
from skcomplex.core.specs import StateSpec, VisualSpec


def _deviations(kind: str, source: object, **settings: object) -> LargeDeviationsResult:
    """Return the large deviation analysis of a source for one kind, run."""
    return large_deviations(source, kind=kind, **settings)


def _mean_field_magnetizations(size: int, beta: float, field: float, samples: int, seed: int) -> np.ndarray:
    """Return exact samples of the magnetization per spin of the mean-field Ising system."""
    totals = np.arange(-size, size + 1, 2)
    logs = (
        special.gammaln(size + 1) - special.gammaln((size + totals) / 2 + 1) - special.gammaln((size - totals) / 2 + 1)
    )
    logs = logs + beta * totals**2 / (2 * size) + beta * field * totals
    weights = np.exp(logs - logs.max())
    return np.random.default_rng(seed).choice(totals, samples, p=weights / weights.sum()) / size


def test_tilted_and_block_generating_functions_of_poisson_counting():
    """Test that both generating functions of a Poisson count with rate 2 equal 2 (e^k - 1) and the rate at 3."""
    tilts = np.linspace(-1, 1, 9)
    exact = 2 * (np.exp(tilts) - 1)
    tilted = _deviations('tilted', [[2.0]], weights=[[1.0]], tilts=tilts).scgf
    np.testing.assert_allclose(tilted, exact, rtol=0.02, atol=1e-12)
    counts = np.random.default_rng(0).poisson(0.2, 10_000_000)
    blocks = _deviations('block', counts, tilts=tilts, block=10, dt=0.1).scgf
    np.testing.assert_allclose(blocks, exact, rtol=0.02, atol=0.005)
    grid = np.linspace(-3, 3, 6001)
    rate = _deviations('tilted', [[2.0]], weights=[[1.0]], tilts=grid, values=3.0).rate_function[0]
    assert rate == pytest.approx(3 * math.log(3 / 2) - 1, rel=0.02)


def test_tilted_generating_function_gallavotti_cohen_symmetry():
    """Test that the current of the driven ring satisfies lambda(k) = lambda(-k - ln 2)."""
    ring = 2 * np.roll(np.eye(5), 1, axis=1) + np.roll(np.eye(5), -1, axis=1)
    current = np.roll(np.eye(5), 1, axis=1) - np.roll(np.eye(5), -1, axis=1)
    tilts = np.linspace(-2, 1, 13)
    np.testing.assert_allclose(
        _deviations('tilted', ring, weights=current, tilts=tilts).scgf,
        _deviations('tilted', ring, weights=current, tilts=-tilts - math.log(2)).scgf,
        atol=1e-8,
    )


def test_histogram_rate_function_mean_field_order_parameter():
    """Test that histograms at sizes 50, 100, and 200 give the exact rate function above the critical temperature."""
    beta = 0.5
    samples = {size: _mean_field_magnetizations(size, beta, 0.0, 10_000_000, size) for size in (50, 100, 200)}
    edges = np.arange(-1.02, 1.03, 0.04)
    rate = _deviations('histogram', samples, bins=edges)
    m = rate.values
    exact = (1 + m) / 2 * np.log1p(m) + (1 - m) / 2 * np.log1p(-m) - beta * m**2 / 2
    inside = (np.abs(m) < 0.5) & np.isfinite(rate.rate_function)
    assert np.max(np.abs(rate.rate_function[inside] - (exact[inside] - exact[inside].min()))) < 0.02
    with pytest.raises(AnalysisError, match='at least 2 sizes'):
        _deviations('histogram', {50: samples[50]})


def test_reweight_magnetization_to_a_field():
    """Test that the magnetization at field 0 reweighted to field 0.05 matches a run at field 0.05."""
    size, beta, field = 100, 0.8, 0.05
    zero = _mean_field_magnetizations(size, beta, 0.0, 2_000_000, 1)
    target = _mean_field_magnetizations(size, beta, field, 2_000_000, 2)
    reweighted = _deviations('reweight', zero, log_weights=beta * field * size * zero)
    assert reweighted.mean == pytest.approx(target.mean(), abs=0.02)
    assert reweighted.effective_size > 1000


def test_block_generating_function_needs_two_blocks():
    """Test that a run shorter than two blocks raises an error."""
    with pytest.raises(AnalysisError, match='fewer than 2 blocks'):
        _deviations('block', [1.0, 2.0], tilts=[0.5], block=5)
    with pytest.raises(AnalysisError, match='needs weights'):
        large_deviations([0, 1, 0, 1], kind='tilted')
    with pytest.raises(AnalysisError, match='does not read block'):
        large_deviations([0, 1, 0, 1], kind='reweight', log_weights=[0.0], block=5)


class CounterElement(Element):
    """Counter element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n

    def _get_laws(self, s):
        """Return the laws of the counter element."""
        return [JumpLaw(1.0, {s.n: 1})]


def test_cloning_estimates_the_poisson_generating_function():
    """Test that cloning a Poisson counter of rate 1 at s = 0.5 is within 2% of `exp(0.5) - 1`."""
    estimate = _deviations(
        'cloning',
        CompositeSystem(elements=[CounterElement('counter')]),
        observable='counter.n',
        tilts=0.5,
        population=300,
        until=20.0,
        random_state=0,
    )
    assert estimate.scgf[0] == pytest.approx(np.exp(0.5) - 1, rel=0.02)
    assert estimate.effective_population.shape == (1,)


def test_cloning_takes_a_function_of_the_segment():
    """Test that a function of each segment's trajectory gives the same estimate as the counter's name."""
    system = CompositeSystem(elements=[CounterElement('counter')])
    settings = {'tilts': -0.5, 'population': 50, 'until': 3.0, 'random_state': 1}
    by_name = _deviations('cloning', system, observable='counter.n', **settings)
    by_function = _deviations(
        'cloning',
        system,
        observable=lambda trajectory: trajectory.get_values('counter.n')[-1] - trajectory.get_values('counter.n')[0],
        **settings,
    )
    assert by_function.scgf == pytest.approx(by_name.scgf)


def test_cloning_warns_of_a_collapsed_population_and_rejects_bad_settings():
    """Test that a strong tilt on few clones warns, and that a duration off the interval grid raises an error."""
    system = CompositeSystem(elements=[CounterElement('counter')])
    with pytest.warns(RuntimeWarning, match='effective population'):
        _deviations('cloning', system, observable='counter.n', tilts=4.0, population=5, until=2.0, random_state=0)
    with pytest.raises(AnalysisError, match='multiple of the interval'):
        large_deviations([0, 1, 0, 1], kind='cloning', observable='counter.n', population=10, interval=1.0, until=2.5)
