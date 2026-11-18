"""Test the kinetic equations of gases and plasmas over position and velocity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt
import pytest
from scipy.optimize import brentq

from skcomplex.core.components.elements import Element
from skcomplex.core.engine.solvers import (
    EulerMaruyamaSolver,
    RK4Solver,
    ScipyODESolver,
    SSPRKSolver,
    StrangSplittingSolver,
)
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.physics.fluids import (
    BGKEquationSystem,
    BoltzmannEquationSystem,
    EnskogSystem,
    RelaxationTimeInteraction,
    VlasovEquationSystem,
)


def sod_density(points: npt.NDArray[np.float64], time: float, gamma: float) -> npt.NDArray[np.float64]:
    """Return the density of the exact Euler solution of the Sod shock tube at given points around the discontinuity."""
    left, right = (1.0, 1.0), (0.125, 0.1)
    sound = [np.sqrt(gamma * pressure / density) for density, pressure in (left, right)]

    def right_jump(pressure: float) -> float:
        factor = np.sqrt((2 / ((gamma + 1) * right[0])) / (pressure + (gamma - 1) / (gamma + 1) * right[1]))
        return (pressure - right[1]) * factor

    def left_fan(pressure: float) -> float:
        return 2 * sound[0] / (gamma - 1) * ((pressure / left[1]) ** ((gamma - 1) / (2 * gamma)) - 1)

    star = brentq(lambda pressure: left_fan(pressure) + right_jump(pressure), 1e-6, 1.0)
    speed = -left_fan(star)
    ratio = (gamma - 1) / (gamma + 1)
    density_left = left[0] * (star / left[1]) ** (1 / gamma)
    density_right = right[0] * (star / right[1] + ratio) / (star / right[1] * ratio + 1)
    shock = (density_right * speed) / (density_right - right[0])
    tail = speed - np.sqrt(gamma * star / density_left)
    head = -sound[0]
    rays = points / time
    fan = left[0] * (2 / (gamma + 1) - (gamma - 1) / ((gamma + 1) * sound[0]) * rays) ** (2 / (gamma - 1))
    return np.select(
        [rays < head, rays < tail, rays < speed, rays < shock],
        [left[0], fan, density_left, density_right],
        right[0],
    )


def amplitude_of(values: npt.NDArray[np.float64], positions: int, velocities: int) -> npt.NDArray[np.float64]:
    """Return the amplitude of the first Fourier mode of the density moment of every frame."""
    density = values.reshape(-1, positions, velocities).sum(axis=2)
    return np.abs(np.fft.rfft(density, axis=1)[:, 1])


def test_bgk_sod_shock_tube_near_the_fluid_limit() -> None:
    """Match the exact Euler solution of the Sod shock tube within 2 percent in the L1 norm at a Knudsen of 1e-4."""
    positions, velocities = 200, 48
    system = BGKEquationSystem(positions=positions, velocities=velocities, knudsen=1e-4)
    solver = StrangSplittingSolver(SSPRKSolver(dt=0.001), RK4Solver(dt=2e-4), dt=0.001)
    trajectory = system.set_params(solver=solver).prepare().simulate(dt=0.2)
    values = trajectory.get_values('gas.x')[-1].reshape(positions, velocities)
    width = 12.0 / velocities
    density = values.sum(axis=1) * width
    points = (np.arange(positions) + 0.5) / positions - 0.5
    exact = sod_density(points, 0.2, gamma=3.0)
    assert np.abs(density - exact).sum() / np.abs(exact).sum() < 0.02


def test_vlasov_default_system_damps_at_the_landau_rate() -> None:
    """Measure a damping rate of the electric energy within 5 percent of 0.1533 with the default set-up."""
    positions, velocities = 32, 64
    trajectory = VlasovEquationSystem().prepare().simulate()
    amplitude = amplitude_of(np.asarray(trajectory.get_values('gas.x')), positions, velocities)
    times = np.asarray(trajectory.times)
    peaks = [
        index
        for index in range(1, times.size - 1)
        if amplitude[index] > amplitude[index - 1] and amplitude[index] > amplitude[index + 1]
    ]
    assert abs(-np.polyfit(times[peaks], np.log(amplitude[peaks]), 1)[0] - 0.1533) < 0.05 * 0.1533


def test_vlasov_two_stream_grows_and_unknown_setup_raises() -> None:
    """Grow the density wave of two beams, and raise a named error for an unknown set-up."""
    positions, velocities = 32, 64
    system = VlasovEquationSystem(positions, velocities, setup='two-stream', amplitude=0.001)
    trajectory = system.prepare().simulate(until=40.0)
    amplitude = amplitude_of(np.asarray(trajectory.get_values('gas.x')), positions, velocities)
    assert amplitude.max() > 10 * amplitude[0]
    with pytest.raises(ValidationError, match='Unknown set-up'):
        VlasovEquationSystem(setup='other').prepare()


def test_boltzmann_conserves_density() -> None:
    """Keep the total of a Boltzmann gas within 1e-10 relative error under streaming and collisions."""
    trajectory = BoltzmannEquationSystem().prepare().simulate()
    values = np.asarray(trajectory.get_values('gas.x'))
    assert abs(values[-1].sum() / values[0].sum() - 1.0) < 1e-10
    assert not np.array_equal(values[-1], values[0])


def test_enskog_contact_value_speeds_collisions_at_a_higher_density() -> None:
    """Relax faster than the Boltzmann equation at the same Knudsen number, by the contact value at the density."""
    dilute = EnskogSystem(diameter=1e-6).prepare().simulate(until=0.2)
    dense = EnskogSystem(diameter=0.5).prepare().simulate(until=0.2)
    start = np.asarray(dilute.get_values('gas.x'))[0]
    assert (
        np.abs(np.asarray(dense.get_values('gas.x'))[-1] - start).max()
        >= np.abs(
            np.asarray(dilute.get_values('gas.x'))[-1] - start,
        ).max()
    )


def test_run_reaches_the_ornstein_uhlenbeck_variance():
    """Test that goal 0, time `1 / theta`, and noise `sigma` give the variance `sigma**2 / (2 theta)` within 5%."""
    theta, sigma = 2.0, 0.6
    process = RelaxationTimeInteraction('ou', target='walkers', state='n', goal=0.0, time=1 / theta, noise=sigma)
    system = System(
        elements=[Element('walkers', values={'n': np.zeros(20_000)}, size=20_000)],
        interactions=[process],
        solver=EulerMaruyamaSolver(dt=0.005),
    )
    values = system.prepare().simulate(until=6, dt=0.5, random_state=0).get_values('walkers.n')[-4:]
    assert values.var(axis=1).mean() == pytest.approx(sigma**2 / (2 * theta), rel=0.05)
    assert values.mean() == pytest.approx(0.0, abs=0.01)


def test_run_relaxes_towards_a_goal_of_the_namespace():
    """Test that a relaxation towards the moving goal `t` follows `t - time (1 - exp(-t / time))`."""
    system = System(
        elements=[Element('belief', values={'x': 0.0})],
        interactions=[RelaxationTimeInteraction('adaptation', target='belief', goal=lambda s: s.t, time=2.0)],
        solver=ScipyODESolver(rtol=1e-10, atol=1e-12),
    )
    trajectory = system.prepare().simulate(until=3.0, dt=0.5)
    expected = trajectory.times - 2.0 * (1 - np.exp(-trajectory.times / 2.0))
    np.testing.assert_allclose(trajectory.get_values('belief.x'), expected, rtol=1e-7, atol=1e-9)


def test_a_goal_of_a_number_pulls_every_component_of_a_vector():
    """Test that a goal given as a number pulls every component of a vector state towards it."""
    system = System(
        elements=[Element('point', values={'x': np.array([0.0, 2.0])})],
        interactions=[RelaxationTimeInteraction('relaxation', target='point', goal=1.0, time=0.1)],
    )
    assert system.prepare().simulate(until=5.0, dt=5.0).get_values('point.x')[-1] == pytest.approx([1.0, 1.0])


def test_prepare_rejects_a_goal_that_is_not_a_function():
    """Test that `prepare` rejects a goal that is neither callable nor a number."""
    with pytest.raises(ValidationError, match='goal'):
        RelaxationTimeInteraction('ou', target='walkers', goal='zero').prepare()
