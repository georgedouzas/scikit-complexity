"""Test the mean squared displacement of particles, its exponent, and the diffusion coefficient."""

import dataclasses
import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import NN

from skcomplex.core.analysis import DisplacementResult, displacement
from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import BaseLaw, RuleLaw
from skcomplex.core.components.spaces import ContinuousSpace, LatticeSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import AgentView, SystemView
from skcomplex.core.errors import AnalysisError
from skcomplex.core.specs import StateSpec, VisualSpec
from skcomplex.systems.biology.evolution import ReplicatorDynamicsSystem


def _displacement(source: object, **settings: object) -> DisplacementResult:
    """Return the displacement analysis of a source, run."""
    return displacement(source, **settings)


def test_displacement_keeps_its_reference_values():
    """Test that the displacement, its exponent, and the coefficient keep values from the previous library."""
    positions = np.sin(np.arange(64.0) ** 2).reshape(16, 2, 2).cumsum(axis=0)
    displacement = _displacement(positions, dt=0.5)
    assert displacement.lags == pytest.approx(np.arange(1, 16) * 0.5)
    assert displacement.msd == pytest.approx(
        [
            1.0031513251316264,
            1.9550222343314765,
            2.938403425767497,
            4.116430615596165,
            5.416999163702931,
            6.441473853372674,
            7.322583537306857,
            8.143335872589564,
            8.621029165730512,
            10.608196796280316,
            11.56209546442886,
            12.547397205312393,
            10.622239087685724,
            9.641781836025046,
            13.678466849526968,
        ],
        rel=1e-9,
    )
    assert displacement.exponent == pytest.approx(0.9765717669094003, rel=1e-9)
    assert displacement.diffusion == pytest.approx(0.48381302515896735, rel=1e-9)


def test_displacement_refuses_states_that_are_no_positions():
    """Test that a group whose states are series has no displacement, and the error names what it holds."""
    trajectory = simulate(ReplicatorDynamicsSystem(), until=1.0, random_state=0)
    with pytest.raises(AnalysisError, match='Its data kinds are: series'):
        _displacement(trajectory)


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

    def step(self, agent: AgentView, system: SystemView) -> None:
        """Move to a random neighbouring cell."""
        cells = agent.neighbor_cells()
        agent.move_to(cells[int(system.rng.integers(len(cells)))])
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

    def step(self, agent: AgentView, system: SystemView) -> None:
        """Take a Gaussian step and wrap into the square."""
        position = np.asarray(agent.location, dtype=float) + system.rng.standard_normal(2)
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
        trajectory = simulate(system, until=600, dt=1, random_state=0)
        displacement = _displacement(trajectory, fit_range=(1, 150))
        assert displacement.exponent == pytest.approx(1, abs=0.05)
        assert displacement.msd[-1] > 20.0**2


def test_mean_squared_displacement_active_particles():
    """Test the ballistic and diffusive regimes of active Brownian particles and their long-time diffusion."""
    positions = _active_brownian(np.random.default_rng(7), 400, 1.0, 1.0)
    ballistic = _displacement(positions, dt=0.01, fit_range=(0.01, 0.1))
    diffusive = _displacement(positions, dt=0.01, fit_range=(10.0, 25.0))
    assert ballistic.exponent == pytest.approx(2, abs=0.1)
    assert diffusive.exponent == pytest.approx(1, abs=0.1)
    assert diffusive.diffusion == pytest.approx(0.5, rel=0.1)


def test_mean_squared_displacement_brownian_motion():
    """Test that 1000 Brownian particles give an exponent within 0.05 of 1 and their diffusion coefficient."""
    steps = np.random.default_rng(3).normal(0, np.sqrt(2 * 0.5 * 0.1), (2000, 1000, 2))
    displacement = _displacement(np.cumsum(steps, axis=0), dt=0.1)
    assert displacement.exponent == pytest.approx(1, abs=0.05)
    assert displacement.diffusion == pytest.approx(0.5, rel=0.05)
    assert displacement.lags.size == 1999


def test_brownian_motion_diffuses_normally():
    """Test that 1000 Brownian particles have a mean squared displacement exponent within 0.05 of 1."""
    walks = np.cumsum(np.random.default_rng(5).standard_normal((2000, 1000, 2)), axis=0)
    assert _displacement(walks).exponent == pytest.approx(1.0, abs=0.05)


def test_mean_squared_displacement_at_chosen_lags():
    """Test that chosen lags keep the displacement of the full curve at those lags."""
    walks = np.cumsum(np.random.default_rng(6).standard_normal((200, 50)), axis=0)
    full = _displacement(walks)
    chosen = _displacement(walks, lags=[1, 10, 100])
    assert chosen.lags.tolist() == [1.0, 10.0, 100.0]
    np.testing.assert_array_equal(chosen.msd, full.msd[[0, 9, 99]])


def test_mean_squared_displacement_rejects_an_empty_range_and_bad_settings():
    """Test that a fit range without lags, a negative time step, and a lag of 0 raise errors."""
    with pytest.raises(AnalysisError, match='fit range'):
        _displacement(np.zeros((10, 2)), fit_range=(100.0, 200.0))
    with pytest.raises(AnalysisError, match='positive time step'):
        displacement(np.zeros((10, 2)), dt=-1.0)
    with pytest.raises(AnalysisError, match='1 frame or more'):
        displacement(np.zeros((10, 2)), lags=[0, 1])
    result = _displacement(np.cumsum(np.ones((20, 3)), axis=0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.msd = np.zeros(1)
