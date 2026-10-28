"""Test the mean squared displacement of particles, its exponent, and the diffusion coefficient."""

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import NN
from sklearn.base import clone

from skcomplex.analysis import DisplacementAnalysis
from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.laws import BaseLaw, RuleLaw
from skcomplex.core.spaces import ContinuousSpace, LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import AgentView, ModelView
from skcomplex.errors import AnalysisError
from skcomplex.specs import StateSpec, VisualSpec


def _displacement(source: object, **settings: object) -> DisplacementAnalysis:
    """Return the displacement analysis of a source, run."""
    return DisplacementAnalysis(**settings).check().prepare(source).run()


class GridWalkerElement(Element):
    """Walker that moves to a random neighbouring cell of a periodic grid at every step."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {}),
        *,
        steps: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.steps = steps

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the stepping rule."""
        return [RuleLaw(self.step)]

    def step(self, agent: AgentView, model: ModelView) -> None:
        """Move to a random neighbouring cell."""
        cells = agent.neighbor_cells()
        agent.move_to(cells[int(model.rng.integers(len(cells)))])
        agent.steps += 1


class DiffuserElement(Element):
    """Particle that takes a Gaussian step of unit variance per coordinate in a periodic square at every step."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {}),
        *,
        steps: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.steps = steps

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the stepping rule."""
        return [RuleLaw(self.step)]

    def step(self, agent: AgentView, model: ModelView) -> None:
        """Take a Gaussian step and wrap into the square."""
        position = np.asarray(agent.location, dtype=float) + model.rng.standard_normal(2)
        agent.move_to(tuple(np.mod(position, 20.0)))
        agent.steps += 1


def _active_brownian(rng: np.random.Generator, particles: int, speed: float, rotation: float) -> np.ndarray:
    """Return positions of two-dimensional active Brownian particles every 0.01 over 100 time units."""
    dt, steps = 0.01, 10_000
    angles = rng.uniform(-np.pi, np.pi, particles) + np.cumsum(
        math.sqrt(2 * rotation * dt) * rng.standard_normal((steps, particles)),
        axis=0,
    )
    moves = speed * dt * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    return np.concatenate([np.zeros((1, particles, 2)), np.cumsum(moves, axis=0)])


def test_mean_squared_displacement_diffusion_on_two_kinds_of_space():
    """Test that Brownian particles in a periodic box and walkers on a periodic grid diffuse without a plateau."""
    continuous = System(
        elements=[GroupElement('particles', element=DiffuserElement, size=300)],
        space=ContinuousSpace(extent=(20.0, 20.0), boundary='periodic'),
    )
    grid = System(
        elements=[GroupElement('walkers', element=GridWalkerElement, size=300)],
        space=LatticeSpace((20, 20), boundary='periodic', neighborhood='von_neumann'),
    )
    for system in (continuous, grid):
        trajectory = Simulation(system, dt=1, random_state=0).check().run(until=600).trajectory_
        displacement = _displacement(trajectory, fit_range=(1, 150))
        assert displacement.exponent_ == pytest.approx(1, abs=0.05)
        assert displacement.msd_[-1] > 20.0**2


def test_mean_squared_displacement_active_particles():
    """Test the ballistic and diffusive regimes of active Brownian particles and their long-time diffusion."""
    positions = _active_brownian(np.random.default_rng(7), 400, 1.0, 1.0)
    ballistic = _displacement(positions, dt=0.01, fit_range=(0.01, 0.1))
    diffusive = _displacement(positions, dt=0.01, fit_range=(10.0, 25.0))
    assert ballistic.exponent_ == pytest.approx(2, abs=0.1)
    assert diffusive.exponent_ == pytest.approx(1, abs=0.1)
    assert diffusive.diffusion_ == pytest.approx(0.5, rel=0.1)


def test_mean_squared_displacement_brownian_motion():
    """Test that 1000 Brownian particles give an exponent within 0.05 of 1 and their diffusion coefficient."""
    steps = np.random.default_rng(3).normal(0, np.sqrt(2 * 0.5 * 0.1), (2000, 1000, 2))
    displacement = _displacement(np.cumsum(steps, axis=0), dt=0.1)
    assert displacement.exponent_ == pytest.approx(1, abs=0.05)
    assert displacement.diffusion_ == pytest.approx(0.5, rel=0.05)
    assert displacement.lags_.size == 1999


def test_brownian_motion_diffuses_normally():
    """Test that 1000 Brownian particles have a mean squared displacement exponent within 0.05 of 1."""
    walks = np.cumsum(np.random.default_rng(5).standard_normal((2000, 1000, 2)), axis=0)
    assert _displacement(walks).exponent_ == pytest.approx(1.0, abs=0.05)


def test_mean_squared_displacement_at_chosen_lags():
    """Test that chosen lags keep the displacement of the full curve at those lags."""
    walks = np.cumsum(np.random.default_rng(6).standard_normal((200, 50)), axis=0)
    full = _displacement(walks)
    chosen = _displacement(walks, lags=[1, 10, 100])
    assert chosen.lags_.tolist() == [1.0, 10.0, 100.0]
    np.testing.assert_array_equal(chosen.msd_, full.msd_[[0, 9, 99]])


def test_mean_squared_displacement_rejects_an_empty_range_and_bad_settings():
    """Test that a fit range without lags, a negative time step, and a lag of 0 raise errors."""
    with pytest.raises(AnalysisError, match='fit range'):
        _displacement(np.zeros((10, 2)), fit_range=(100.0, 200.0))
    with pytest.raises(AnalysisError, match='positive time step'):
        DisplacementAnalysis(dt=-1.0).check()
    with pytest.raises(AnalysisError, match='1 frame or more'):
        DisplacementAnalysis(lags=[0, 1]).check()
    assert not hasattr(clone(_displacement(np.cumsum(np.ones((20, 3)), axis=0))), 'msd_')
