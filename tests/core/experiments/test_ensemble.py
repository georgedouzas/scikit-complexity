"""Test the `_ensemble` module of `skcomplex.core.experiments`."""

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import NoiseLaw, RateLaw
from skcomplex.core.engine.solvers import KineticSolver, ScipyODESolver
from skcomplex.core.errors import TagError
from skcomplex.core.experiments import ensemble
from skcomplex.core.systems import System
from skcomplex.systems.biology.epidemiology import SIRSystem
from skcomplex.systems.mathematics.stochastic_processes import PoissonProcessSystem
from skcomplex.systems.physics.statistical_mechanics import CurieWeissSystem


class AmountElement(Element):
    """Element that holds one nonnegative scalar `x`, an amount."""

    def __init__(self, label: str, *, x: Annotated[float, RealSet.closed_open(0, oo)] = 0.0) -> None:
        super().__init__(label)
        self.x = x


def _ornstein_uhlenbeck(s):
    """Return the drift and the noise of a process that relaxes to zero with white noise."""
    x = s.target.x
    return [RateLaw(x.diff(s.t), -x), NoiseLaw(x, 0.5)]


@dataclass(frozen=True)
class PeakResult:
    """Height and time of the peak of one series."""

    peak: float
    peak_time: float


def peak(trajectory):
    """Return the height and time of the peak of the infected."""
    values = np.asarray(trajectory.get_values('I.x'), dtype=float)
    return PeakResult(float(values.max()), float(trajectory.times[values.argmax()]))


def test_analyses_of_stochastic_sir_replicates():
    """Test that 20 stochastic SIR replicates with a peak analysis give one row per run with the peak of that run."""
    result = ensemble(
        SIRSystem(stochastic=True),
        replicates=20,
        analyses=[peak],
        until=50,
        outputs=['I.x'],
        random_state=7,
    )
    rows = result.rows
    assert len(rows) == 20
    for row, series in zip(rows, result.get_series('I.x'), strict=True):
        assert row['peak.peak'] == series.max()
        assert 0 <= row['peak.peak_time'] <= 50
    assert len({row['peak.peak'] for row in rows}) > 1


def test_ensemble_of_ornstein_uhlenbeck_replicates():
    """Test that the ensemble of 200 Ornstein-Uhlenbeck replicates has one row per replicate carrying its seed."""
    process = Interaction('process', roles={'target': 'ou'}, laws=_ornstein_uhlenbeck)
    system = System(elements=[Element('ou', values={'x': 1.0})], interactions=[process])
    result = ensemble(system, replicates=200, until=2, dt=0.1, outputs=['ou.x'], random_state=0)
    times, values, seeds = result.get_ensemble('ou.x')
    assert values.shape == (200, times.size)
    assert seeds.tolist() == list(range(200))
    assert len({tuple(row) for row in values}) == 200
    assert values[:, -1].mean() == pytest.approx(np.exp(-2.0), abs=0.1)


def test_replicates_are_independent_and_reproducible():
    """Test that no two of 20 stochastic replicates are identical, and that the seed reproduces them."""
    first = ensemble(SIRSystem(stochastic=True), replicates=20, until=50, outputs=['I.x'], random_state=7)
    second = ensemble(SIRSystem(stochastic=True), replicates=20, until=50, outputs=['I.x'], random_state=7)
    assert len({tuple(series) for series in first.get_series('I.x')}) == 20
    np.testing.assert_array_equal(first.get_series('I.x'), second.get_series('I.x'))


def test_ensemble_of_a_deterministic_system_names_the_randomness():
    """Test that an ensemble of the deterministic SIR system raises a tag error naming the randomness."""
    with pytest.raises(TagError, match='randomness stochastic'):
        ensemble(SIRSystem(), replicates=2, until=5)


def test_ensemble_of_a_deterministic_system_with_a_solver_that_draws():
    """Test that a solver decides an ensemble of a deterministic system by whether it draws, as KineticSolver does."""
    system = System(elements=[AmountElement('stock', x=1.0)])
    assert not ScipyODESolver.draws
    assert KineticSolver.draws
    with pytest.raises(TagError, match='randomness'):
        ensemble(system, replicates=2, until=1, solver=ScipyODESolver())


def test_ensemble_prepares_once(monkeypatch):
    """Test that an ensemble of 5 runs of a system prepares it once and simulates it with 5 streams."""
    calls = []
    prepare = System._prepare

    def counted(self, context=None):
        calls.append(self.label)
        prepare(self, context)

    monkeypatch.setattr(System, '_prepare', counted)
    result = ensemble(PoissonProcessSystem(3), replicates=5, until=10, outputs=['counters.count'], random_state=0)
    assert len(calls) == 1
    assert len({tuple(series.ravel()) for series in result.get_series('counters.count')}) == 5


def test_ensemble_gives_every_replicate_its_own_system_seed():
    """Test that Curie-Weiss replicates start from different spins and order with both signs, reproducibly."""
    system = CurieWeissSystem(50, temperature=0.5)
    result = ensemble(system, replicates=20, until=20.0, outputs=['spins.spin'], random_state=3)
    again = ensemble(system, replicates=20, until=20.0, outputs=['spins.spin'], random_state=3)
    starts = [run.series['spins.spin'][0] for run in result.runs]
    signs = {float(np.sign(np.sum(run.series['spins.spin'][-1]))) for run in result.runs}
    assert len({tuple(start) for start in starts}) == 20
    assert {-1.0, 1.0} <= signs
    assert result.get_column('random_state').tolist() == again.get_column('random_state').tolist()
    for first, second in zip(result.runs, again.runs, strict=True):
        np.testing.assert_array_equal(first.series['spins.spin'], second.series['spins.spin'])
