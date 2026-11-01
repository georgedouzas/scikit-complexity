"""Test the coarse-graining of agents into fields, and of members into densities over the values of a state."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.analysis import CoarseGrainAnalysis, SpectrumAnalysis
from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.laws.utils import moment
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.system import System
from skcomplex.core.times import ContinuousTime
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import AnalysisError
from skcomplex.models import run
from skcomplex.specs import ObservableSpec, StateSpec, VisualSpec

CLASSES = np.arange(9.0)


class TraderElement(Element):
    """Agent with a wealth."""

    def __init__(self, label: str, visual: VisualSpec | None = None, *, wealth: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.wealth = wealth


class DensityElement(Element):
    """Density of members at one value of a state."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        f: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.f = f


def density(s):
    """Return the density of a distribution."""
    return moment(s.f, 0, CLASSES)


def mean_class(s):
    """Return the mean class of a distribution."""
    return moment(s.f, 1, CLASSES, normalized=True)


def _coarse_grain(positions: object, shape: list[int], **settings: object) -> np.ndarray:
    """Return the field of agents coarse-grained on a grid."""
    return CoarseGrainAnalysis(shape, **settings).check().prepare(positions).run().field_


def test_coarse_grain_density():
    """Test that 10,000 agents coarse-grained on a 32 by 32 grid sum to 10,000 and have a structure factor."""
    rng = np.random.default_rng(6)
    density = _coarse_grain(rng.random((10_000, 2)) * 5, [32, 32], box=[5.0, 5.0])
    assert density.sum() == 10_000
    assert SpectrumAnalysis('structure-factor').check().prepare(density).run().power_.size > 0
    polarization = _coarse_grain(rng.random((100, 2)), [4, 4], box=[1.0, 1.0], values=np.ones((100, 2)))
    assert polarization.shape == (4, 4, 2)
    with pytest.raises(AnalysisError, match='one entry per dimension'):
        CoarseGrainAnalysis([4], box=[1.0, 1.0]).check()


def test_coarse_grain_agents_of_a_trajectory():
    """Test that the walkers of a trajectory give one grid per frame in the box of their space."""
    trajectory = run('run_and_tumble', n_agents=50, settings={'until': 3}).trajectory_
    field = _coarse_grain(trajectory, [5, 5])
    assert field.shape == (4, 5, 5)
    assert np.all(field.sum(axis=(1, 2)) == 50)
    with pytest.raises(AnalysisError, match='needs the sides of the box'):
        CoarseGrainAnalysis([5, 5]).check().prepare(np.zeros((3, 2)))


def test_density_over_a_state():
    """Test that the wealth of 1,000 traders over 20 values gives the fractions nearest every value, summing to 1."""
    wealth = np.random.default_rng(0).exponential(5.0, size=1000)
    values = np.linspace(0.0, 19.0, 20)
    densities = CoarseGrainAnalysis(over=values).check().prepare(wealth).run().field_
    nearest = np.argmin(np.abs(wealth[:, None] - values[None, :]), axis=1)
    np.testing.assert_allclose(densities, np.bincount(nearest, minlength=20) / 1000)
    assert densities.sum() == pytest.approx(1.0)


def test_density_over_a_state_bins_to_the_nearest_value():
    """Test that values between two given values count at the nearest one, and values outside at the closest end."""
    densities = CoarseGrainAnalysis(over=[0.0, 1.0, 2.0]).check().prepare([-1.0, 0.4, 0.6, 1.5, 9.0]).run().field_
    np.testing.assert_allclose(densities, [0.4, 0.4, 0.2])


def test_density_over_a_state_starts_a_distribution():
    """Test that the densities of 1,000 traders start a distribution of density 1 whose mean is the traders' mean."""
    wealth = np.random.default_rng(0).integers(0, 9, size=1000).astype(float)
    densities = CoarseGrainAnalysis(over=CLASSES).check().prepare(wealth).run().field_
    distribution = GroupElement('wealth', element=DensityElement, size=9, on='sites', values={'f': densities})
    system = System(elements=[distribution], space=LatticeSpace(9, boundary='fixed'), time=ContinuousTime())
    observables = [
        ObservableSpec('density', density, group='wealth'),
        ObservableSpec('mean', mean_class, group='wealth'),
    ]
    trajectory = Simulation(system, observables=observables, dt=1.0).check().prepare().run(until=0).trajectory_
    assert trajectory.values('density')[0] == pytest.approx(1.0)
    assert trajectory.values('mean')[0] == pytest.approx(wealth.mean())


def test_density_over_a_state_of_a_trajectory():
    """Test that the members of a group of a trajectory give one density per frame over the values of a state."""
    wealth = np.random.default_rng(1).integers(0, 9, size=50).astype(float)
    traders = GroupElement('traders', element=TraderElement, size=50, values={'wealth': wealth})
    trajectory = Simulation(System(elements=[traders]), dt=1.0).check().prepare().run(until=2).trajectory_
    analysis = CoarseGrainAnalysis(over=CLASSES, state='traders.wealth').check().prepare(trajectory).run()
    assert analysis.field_.shape == (3, 9)
    np.testing.assert_allclose(analysis.field_.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    ('settings', 'message'),
    [
        ({'over': [1.0, 0.0]}, 'increasing vector'),
        ({'over': [0.0, 1.0], 'values': [1.0]}, 'takes no `values`'),
        ({'state': 'traders.wealth'}, 'Set `over`'),
    ],
)
def test_density_over_a_state_rejects_bad_settings(settings, message):
    """Test that the analysis rejects values that are not increasing, weights, and a state without values."""
    with pytest.raises(AnalysisError, match=message):
        CoarseGrainAnalysis(**settings).check()
