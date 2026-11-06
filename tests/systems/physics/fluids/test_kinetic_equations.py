"""Test the kinetic equations of gases and plasmas over position and velocity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt
import pytest
from scipy.optimize import brentq

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import RK4Solver, SSPRKSolver, StrangSplittingSolver
from skcomplex.core.errors import ValidationError
from skcomplex.systems import create_system, describe
from skcomplex.systems.physics.fluids import enskog, vlasov_equation


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
    system = create_system('bgk_equation', positions=positions, velocities=velocities, knudsen=1e-4)
    solver = StrangSplittingSolver(SSPRKSolver(dt=0.001), RK4Solver(dt=2e-4), dt=0.001)
    trajectory = Simulation(system, dt=0.2, solver=solver).prepare().run(until=0.2).trajectory_
    values = trajectory.values('gas.f')[-1].reshape(positions, velocities)
    width = 12.0 / velocities
    density = values.sum(axis=1) * width
    points = (np.arange(positions) + 0.5) / positions - 0.5
    exact = sod_density(points, 0.2, gamma=3.0)
    assert np.abs(density - exact).sum() / np.abs(exact).sum() < 0.02


def test_vlasov_default_system_damps_at_the_landau_rate() -> None:
    """Measure a damping rate of the electric energy within 5 percent of 0.1533 with the default set-up."""
    positions, velocities = 32, 64
    defaults = describe('vlasov_equation').defaults
    settings = {key: value for key, value in defaults.items() if key != 'until'}
    trajectory = (
        Simulation(create_system('vlasov_equation'), **settings).prepare().run(until=defaults['until']).trajectory_
    )
    amplitude = amplitude_of(np.asarray(trajectory.values('gas.f')), positions, velocities)
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
    defaults = describe('vlasov_equation').defaults
    settings = {key: value for key, value in defaults.items() if key != 'until'}
    system = vlasov_equation(positions, velocities, setup='two-stream', amplitude=0.001)
    trajectory = Simulation(system, **settings).prepare().run(until=40.0).trajectory_
    amplitude = amplitude_of(np.asarray(trajectory.values('gas.f')), positions, velocities)
    assert amplitude.max() > 10 * amplitude[0]
    with pytest.raises(ValidationError, match='Unknown set-up'):
        vlasov_equation(setup='other')


def test_boltzmann_conserves_density() -> None:
    """Keep the total of a Boltzmann gas within 1e-10 relative error under streaming and collisions."""
    defaults = describe('boltzmann_equation').defaults
    settings = {key: value for key, value in defaults.items() if key != 'until'}
    trajectory = (
        Simulation(create_system('boltzmann_equation'), **settings).prepare().run(until=defaults['until']).trajectory_
    )
    values = np.asarray(trajectory.values('gas.f'))
    assert abs(values[-1].sum() / values[0].sum() - 1.0) < 1e-10
    assert not np.array_equal(values[-1], values[0])


def test_enskog_contact_value_speeds_collisions_at_a_higher_density() -> None:
    """Relax faster than the Boltzmann equation at the same Knudsen number, by the contact value at the density."""
    defaults = describe('enskog').defaults
    settings = {key: value for key, value in defaults.items() if key != 'until'}
    dilute = Simulation(enskog(diameter=1e-6), **settings).prepare().run(until=0.2).trajectory_
    dense = Simulation(enskog(diameter=0.5), **settings).prepare().run(until=0.2).trajectory_
    start = np.asarray(dilute.values('gas.f'))[0]
    assert (
        np.abs(np.asarray(dense.values('gas.f'))[-1] - start).max()
        >= np.abs(
            np.asarray(dilute.values('gas.f'))[-1] - start,
        ).max()
    )
