"""Test the `_ensemble` module of `skcomplex.core.experiments`."""

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import CompartmentElement, Element
from skcomplex.core.components.laws import NoiseLaw, RateLaw
from skcomplex.core.components.systems import System
from skcomplex.core.engine.solvers import KineticSolver, ScipyODESolver
from skcomplex.core.errors import TagError
from skcomplex.core.experiments import ensemble
from skcomplex.core.specs import StateSpec, VisualSpec
from skcomplex.systems.biology.epidemiology import SIRSystem


class OrnsteinUhlenbeckElement(Element):
    """Ornstein-Uhlenbeck process that relaxes to zero with white noise."""

    def __init__(self, label: str, visual: VisualSpec | None = None, *, x: Annotated[object, StateSpec()] = 1.0):
        super().__init__(label, visual=visual)
        self.x = x

    def _get_laws(self, s):
        """Return the drift and the noise of the process."""
        return [RateLaw(s.x, -s.x), NoiseLaw(s.x, 0.5)]


@dataclass(frozen=True)
class PeakResult:
    """Height and time of the peak of one series."""

    peak: float
    peak_time: float


def peak(trajectory):
    """Return the height and time of the peak of the infected."""
    values = np.asarray(trajectory.values('I.n'), dtype=float)
    return PeakResult(float(values.max()), float(trajectory.times[values.argmax()]))


def test_analyses_of_stochastic_sir_replicates():
    """Test that 20 stochastic SIR replicates with a peak analysis give one row per run with the peak of that run."""
    result = ensemble(
        SIRSystem(stochastic=True),
        replicates=20,
        analyses=[peak],
        until=50,
        outputs=['I.n'],
        random_state=7,
    )
    rows = result.rows
    assert len(rows) == 20
    for row, series in zip(rows, result.series('I.n'), strict=True):
        assert row['peak.peak'] == series.max()
        assert 0 <= row['peak.peak_time'] <= 50
    assert len({row['peak.peak'] for row in rows}) > 1


def test_ensemble_of_ornstein_uhlenbeck_replicates():
    """Test that the ensemble of 200 Ornstein-Uhlenbeck replicates has one row per replicate carrying its seed."""
    system = System(elements=[OrnsteinUhlenbeckElement('ou')])
    result = ensemble(system, replicates=200, until=2, dt=0.1, outputs=['ou.x'], random_state=0)
    times, values, seeds = result.ensemble('ou.x')
    assert values.shape == (200, times.size)
    assert seeds.tolist() == list(range(200))
    assert len({tuple(row) for row in values}) == 200
    assert values[:, -1].mean() == pytest.approx(np.exp(-2.0), abs=0.1)


def test_replicates_are_independent_and_reproducible():
    """Test that no two of 20 stochastic replicates are identical, and that the seed reproduces them."""
    first = ensemble(SIRSystem(stochastic=True), replicates=20, until=50, outputs=['I.n'], random_state=7)
    second = ensemble(SIRSystem(stochastic=True), replicates=20, until=50, outputs=['I.n'], random_state=7)
    assert len({tuple(series) for series in first.series('I.n')}) == 20
    np.testing.assert_array_equal(first.series('I.n'), second.series('I.n'))


def test_ensemble_of_a_deterministic_system_names_the_randomness():
    """Test that an ensemble of the deterministic SIR system raises a tag error naming the randomness."""
    with pytest.raises(TagError, match='randomness stochastic'):
        ensemble(SIRSystem(), replicates=2, until=5)


def test_ensemble_of_a_deterministic_system_with_a_solver_that_draws():
    """Test that a solver decides an ensemble of a deterministic system by whether it draws, as KineticSolver does."""
    system = System(elements=[CompartmentElement('stock', n=1.0)])
    assert not ScipyODESolver.draws
    assert KineticSolver.draws
    with pytest.raises(TagError, match='randomness'):
        ensemble(system, replicates=2, until=1, solver=ScipyODESolver())
