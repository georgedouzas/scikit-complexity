"""Test the physics models of mechanics, orbits, oscillators, synchronization, waves, heat, and cosmology."""

import math

import numpy as np
import pytest

from skcomplex.core.compile import compile_system
from skcomplex.core.model import PairInteraction
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.models.physics.celestial_mechanics import (
    n_body,
    restricted_three_body,
    sitnikov,
    stormer_problem,
    two_body_orbit,
)
from skcomplex.models.physics.cosmology import friedmann_expansion
from skcomplex.models.physics.heat import cooling_coffee
from skcomplex.models.physics.mechanics import (
    ballistic_pendulum,
    cart_pendulum,
    coupled_pendulums,
    double_pendulum,
    gimballed_gyroscope,
    huygens_clocks,
    magnetic_pendulum,
    parametric_pendulum,
    particle_in_a_well,
    rotating_balls_on_rods,
    rotating_swing,
    two_mass_spring_damper,
)
from skcomplex.models.physics.nonlinear_oscillators import (
    adler_equation,
    billiards,
    forced_van_der_pol,
    stuart_landau,
)
from skcomplex.models.physics.synchronization import (
    aggregate_averaging_synchronization,
    coupled_rossler,
    kuramoto,
    kuramoto_sakaguchi_chimera,
    mirollo_strogatz,
    nonlocal_ring_chimera,
    ott_antonsen,
    winfree,
)
from skcomplex.models.physics.waves import schrodinger, spring_chain, standing_waves

ACCURATE = ScipyODE(method='DOP853', rtol=1e-10, atol=1e-12)
G = 9.81


def _run(system, until, dt, solver=ACCURATE, **settings):
    """Return the trajectory of a system run to a time."""
    return Simulation(system, dt=dt, solver=solver, **settings).run(until=until).trajectory_


def _series(trajectory, *names):
    """Return the series of several variables as arrays."""
    return [np.asarray(trajectory[name], dtype=float) for name in names]


def _upward_crossings(times, values):
    """Return the interpolated times where a series crosses zero upwards."""
    index = np.flatnonzero((values[:-1] < 0) & (values[1:] >= 0))
    fraction = -values[index] / (values[index + 1] - values[index])
    return times[index] + fraction * (times[index + 1] - times[index])


def _relative_spread(values):
    """Return the range of a conserved quantity relative to its size."""
    return np.ptp(values) / max(np.abs(values).max(), 1e-12)


def test_kepler_orbit_conserves_energy_and_momentum_and_follows_the_third_law():
    """Test that the Kepler orbit keeps its energy and angular momentum for ten periods with Kepler's period."""
    eccentricity, G_, mass, radius = 0.5, 1.0, 1000.0, 10.0
    axis = radius / (1 - eccentricity)
    period = 2 * math.pi * math.sqrt(axis**3 / (G_ * (mass + 1e-6)))
    trajectory = _run(two_body_orbit(eccentricity, G=G_, mass=mass, radius=radius), 10 * period, period / 100)
    star, planet = _series(trajectory, 'star.x', 'planet.x')
    star_v, planet_v = _series(trajectory, 'star.v', 'planet.v')
    relative, velocity = planet - star, planet_v - star_v
    distance = np.linalg.norm(relative, axis=1)
    energy = 0.5 * mass * (star_v**2).sum(1) + 0.5 * 1e-6 * (planet_v**2).sum(1) - G_ * mass * 1e-6 / distance
    momentum = sum(
        weight * (position[:, 0] * speed[:, 1] - position[:, 1] * speed[:, 0])
        for weight, position, speed in ((mass, star, star_v), (1e-6, planet, planet_v))
    )
    assert _relative_spread(energy) < 1e-7
    assert _relative_spread(momentum) < 1e-7
    crossings = _upward_crossings(trajectory.times, relative[:, 1])
    assert len(crossings) >= 9
    assert np.diff(crossings).mean() == pytest.approx(period, rel=1e-3)
    assert np.ptp(np.linalg.norm(velocity, axis=1)) > 0


def test_double_pendulum_conserves_energy():
    """Test that the double pendulum keeps its energy while it moves chaotically."""
    trajectory = _run(double_pendulum(), 10.0, 0.05)
    t1, t2, w1, w2 = _series(
        trajectory,
        'pendulum.theta1',
        'pendulum.theta2',
        'pendulum.theta1_d1',
        'pendulum.theta2_d1',
    )
    kinetic = 0.5 * 2 * w1**2 + 0.5 * (w2**2 + 2 * w1 * w2 * np.cos(t1 - t2))
    potential = -2 * G * np.cos(t1) - G * np.cos(t2)
    assert np.ptp(kinetic + potential) < 1e-6


def test_coupled_pendulums_oscillate_in_their_antiphase_mode():
    """Test that identical coupled pendulums started in antiphase oscillate at `sqrt(g / l + 2 k / m)`."""
    trajectory = _run(coupled_pendulums(0.01, -0.01, k=0.5), 20.0, 0.05)
    left, right = (np.sin(angle) for angle in _series(trajectory, 'left.theta', 'right.theta'))
    crossings = _upward_crossings(trajectory.times, left)
    assert np.allclose(left, -right, atol=1e-9)
    assert np.diff(crossings).mean() == pytest.approx(2 * math.pi / math.sqrt(G + 1.0), rel=1e-4)


def test_cart_pendulum_conserves_horizontal_momentum():
    """Test that the cart and the pendulum keep their horizontal momentum without a force."""
    trajectory = _run(cart_pendulum(1.0, m1=2.0, m2=1.0), 10.0, 0.05)
    velocity, theta, omega = _series(trajectory, 'cart.y_d1', 'cart.theta', 'cart.theta_d1')
    momentum = 3.0 * velocity + omega * np.cos(theta)
    assert np.abs(momentum).max() < 1e-8
    assert np.ptp(velocity) > 0.1


def test_parametric_pendulum_resonates_at_twice_its_frequency():
    """Test that small oscillations grow when the pivot oscillates at twice the natural frequency."""
    resonant = _run(parametric_pendulum(0.01, epsilon=0.3), 20.0, 0.05)
    detuned = _run(parametric_pendulum(0.01, epsilon=0.3, Omega=3.3 * math.sqrt(G)), 20.0, 0.05)
    assert np.abs(np.sin(_series(resonant, 'swing.theta')[0])).max() > 0.3
    assert np.abs(np.sin(_series(detuned, 'swing.theta')[0])).max() < 0.03


def test_magnetic_pendulum_comes_to_rest_near_a_magnet():
    """Test that the damped magnetic pendulum stops near one of the magnets."""
    trajectory = _run(magnetic_pendulum(), 100.0, 5.0, solver=ScipyODE())
    position, velocity = (array[-1] for array in _series(trajectory, 'bob.x', 'bob.v'))
    magnets = np.stack([np.cos(2 * np.pi * np.arange(3) / 3), np.sin(2 * np.pi * np.arange(3) / 3)], axis=1)
    assert np.linalg.norm(velocity) < 1e-3
    assert np.linalg.norm(magnets - position, axis=1).min() < 0.5


def test_ballistic_pendulum_rises_to_the_height_of_momentum_conservation():
    """Test that the pendulum hit by the bullet rises to `(m v / (m + M))² / (2 g)`."""
    trajectory = _run(ballistic_pendulum(speed=200.0), 2.0, 0.002)
    theta = _series(trajectory, 'pendulum.theta')[0]
    height = 1 - np.cos(theta).min()
    assert trajectory.events[0].time == pytest.approx(1.0 / 200.0)
    assert height == pytest.approx((0.01 * 200 / 1.01) ** 2 / (2 * G), rel=1e-4)


def test_huygens_clocks_synchronize_in_antiphase():
    """Test that the default clocks run quickly past the escapement switch and lock in antiphase."""
    trajectory = Simulation(huygens_clocks(), dt=0.05).run(until=80.0).trajectory_
    late = trajectory.times >= 60.0
    first, second = trajectory['clock_0.theta'][late], trajectory['clock_1.theta'][late]
    assert np.corrcoef(first, second)[0, 1] < -0.95


def test_huygens_pendulums_without_escapement_conserve_energy():
    """Test that pendulums on undamped supports joined by a spring keep their total energy."""
    system = huygens_clocks(2, theta=0.3, k=2.0, M=0.5, rho=0.0, lam=0.0)
    trajectory = _run(system, 10.0, 0.05)
    t1, t2, x1, x2 = _series(trajectory, 'clock_0.theta', 'clock_1.theta', 'clock_0.x', 'clock_1.x')
    w1, w2, v1, v2 = _series(trajectory, 'clock_0.theta_d1', 'clock_1.theta_d1', 'clock_0.x_d1', 'clock_1.x_d1')
    kinetic = sum(0.5 * (w**2 + v**2 + 2 * v * w * np.cos(t)) + 0.25 * v**2 for t, w, v in ((t1, w1, v1), (t2, w2, v2)))
    potential = -9.8 * (np.cos(t1) + np.cos(t2)) + (x2 - x1) ** 2
    assert np.ptp(kinetic + potential) < 1e-6
    assert np.abs(t2).max() > 0.05


def test_two_mass_spring_damper_conserves_energy_without_damping():
    """Test that the two masses between walls keep their energy without dampers and forces."""
    trajectory = _run(two_mass_spring_damper(k=2.0), 20.0, 0.1)
    x1, x2, v1, v2 = (array[:, 0] for array in _series(trajectory, 'first.x', 'second.x', 'first.v', 'second.v'))
    energy = 0.5 * (v1**2 + v2**2) + 0.5 * (x1 - 1) ** 2 + 0.5 * (x2 - 2) ** 2 + (x2 - x1 - 1) ** 2
    assert np.ptp(energy) < 1e-8


@pytest.mark.parametrize(('v', 'bounded'), [(0.5, True), (1.5, False)])
def test_particle_in_a_well_is_bound_below_the_barrier(v, bounded):
    """Test that the particle stays in the well below the barrier energy `4 k e^-2` and escapes above it."""
    trajectory = _run(particle_in_a_well(1.0, v), 20.0, 0.05)
    x, velocity = (array[:, 0] for array in _series(trajectory, 'particle.x', 'particle.v'))
    energy = 0.5 * velocity**2 + 2 * x**2 * np.exp(-x)
    assert np.ptp(energy) < 1e-8
    assert (x.max() < 2) == bounded


def test_rotating_swing_conserves_angular_momentum():
    """Test that the swing without torque keeps its angular momentum while the chairs swing out."""
    trajectory = _run(rotating_swing(), 10.0, 0.05)
    theta, omega = _series(trajectory, 'swing.theta', 'swing.phi_d1')
    momentum = (10.0 + 8 * (1 + np.sin(theta)) ** 2) * omega
    assert _relative_spread(momentum) < 1e-8
    assert np.ptp(omega) > 0.1


def test_rotating_balls_turn_faster_when_pulled_in():
    """Test that halving the radius of the balls quadruples their angular velocity."""
    radius = {'right.r': lambda t: 1.0 if t < 1 else 0.5}
    trajectory = _run(rotating_balls_on_rods(1.0, r=1.0), 2.0, 0.1, schedule=radius)
    theta = np.unwrap(_series(trajectory, 'right.theta')[0])
    early, late = np.diff(theta[:10]).mean(), np.diff(theta[-5:]).mean()
    assert late / early == pytest.approx(4.0)


def test_gimballed_gyroscope_precesses_at_torque_over_angular_momentum():
    """Test that the mean precession rate is `-torque / (C spin cos beta)`, shrinking as the rotor spins faster."""
    for spin in (25.0, 50.0, 100.0):
        trajectory = _run(gimballed_gyroscope(spin=spin), 10.0, 0.01)
        mean_rate = trajectory['gyroscope.alpha'][-1] / 10.0
        assert mean_rate == pytest.approx(-1.0 / (2.0 * spin * np.cos(0.3)), rel=0.01)


def test_gimballed_gyroscope_conserves_its_cyclic_momenta_and_precesses():
    """Test that the gyroscope keeps the momenta of its outer ring and rotor and precesses under the torque."""
    trajectory = _run(gimballed_gyroscope(), 5.0, 0.05)
    beta, d_alpha, d_gamma = _series(trajectory, 'gyroscope.beta', 'gyroscope.alpha_d1', 'gyroscope.gamma_d1')
    inertia = 1.0 + 1.5 * np.cos(beta) ** 2 + 2.5 * np.sin(beta) ** 2
    outer = inertia * d_alpha + 2.0 * d_gamma * np.sin(beta)
    rotor = 2.0 * (d_alpha * np.sin(beta) + d_gamma)
    assert np.ptp(outer) < 1e-6
    assert np.ptp(rotor) < 1e-6
    assert np.abs(d_alpha).max() > 0.005
    assert np.ptp(beta) < 0.01


def test_figure_eight_returns_after_its_period():
    """Test that the three bodies of the figure eight come back to their places after one period."""
    trajectory = _run(n_body(), 6.32591398, 6.32591398 / 100)
    for index in range(3):
        position = _series(trajectory, f'body_{index}.x')[0]
        assert np.allclose(position[-1], position[0], atol=1e-4)
    momentum = sum(_series(trajectory, f'body_{index}.v')[0] for index in range(3))
    assert np.abs(momentum).max() < 1e-8


def test_inner_solar_system_keeps_the_earth_on_its_orbit():
    """Test that the Earth of the inner solar system preset goes around the Sun once a year."""
    trajectory = _run(n_body('inner_solar_system'), 1.0, 0.01)
    sun, earth = _series(trajectory, 'body_0.x', 'body_3.x')
    assert np.linalg.norm(earth - sun, axis=1) == pytest.approx(1.0, rel=1e-3)
    assert np.allclose(earth[-1] - sun[-1], [1.0, 0.0], atol=0.01)


def test_restricted_three_body_conserves_the_jacobi_constant():
    """Test that the light body in the rotating frame keeps its Jacobi constant."""
    mu = 0.01215
    trajectory = _run(restricted_three_body(mu=mu), 20.0, 0.02)
    x, y, vx, vy = _series(trajectory, 'body.x', 'body.y', 'body.x_d1', 'body.y_d1')
    jacobi = x**2 + y**2 + 2 * (1 - mu) / np.hypot(x + mu, y) + 2 * mu / np.hypot(x - 1 + mu, y) - vx**2 - vy**2
    assert np.ptp(jacobi) < 1e-8


def test_restricted_three_body_defaults_are_chaotic():
    """Test that the default light body visits the Moon and separates from a neighbour started 1e-8 away."""
    mu = 0.01215
    first = _run(restricted_three_body(), 20.0, 0.02)
    second = _run(restricted_three_body(x=0.5 + 1e-8), 20.0, 0.02)
    x, y = _series(first, 'body.x', 'body.y')
    other_x, other_y = _series(second, 'body.x', 'body.y')
    assert np.hypot(x - 1 + mu, y).min() < 0.2
    assert np.hypot(x - other_x, y - other_y)[-1] > 1e-6


def test_stormer_problem_conserves_speed_and_axial_momentum():
    """Test that a charge in a dipole field keeps its speed and its canonical angular momentum about the axis."""
    trajectory = _run(stormer_problem(), 20.0, 0.02)
    x, y, z, vx, vy, vz = _series(trajectory, *(f'charge.{name}' for name in ('x', 'y', 'z', 'x_d1', 'y_d1', 'z_d1')))
    radius = np.sqrt(x**2 + y**2 + z**2)
    assert np.ptp(vx**2 + vy**2 + vz**2) < 1e-8
    assert np.ptp(x * vy - y * vx + (x**2 + y**2) / radius**3) < 1e-8
    assert radius.max() < 2


def test_stuart_landau_settles_on_its_limit_cycle():
    """Test that the Stuart-Landau oscillator settles on the circle of radius `sqrt(mu)`."""
    trajectory = _run(stuart_landau(mu=0.5, beta=0.5), 40.0, 0.1)
    x, y = _series(trajectory, 'oscillator.x', 'oscillator.y')
    phase = np.unwrap(np.arctan2(y, x))
    assert np.hypot(x, y)[-1] == pytest.approx(math.sqrt(0.5), rel=1e-6)
    assert np.diff(phase[-20:]).mean() / 0.1 == pytest.approx(1.0 - 0.25, rel=1e-4)


def test_forced_van_der_pol_locks_to_a_strong_force():
    """Test that a strong force entrains the Van der Pol oscillator to its own frequency."""
    trajectory = _run(forced_van_der_pol(A=1.0, omega=1.1), 120.0, 0.05, solver=ScipyODE(rtol=1e-8))
    crossings = _upward_crossings(trajectory.times, _series(trajectory, 'oscillator.x')[0])
    assert np.diff(crossings[-5:]).mean() == pytest.approx(2 * math.pi / 1.1, rel=1e-3)


@pytest.mark.parametrize('detuning', [0.5, 1.5])
def test_adler_equation_locks_inside_the_tongue_and_slips_outside(detuning):
    """Test that the phase difference locks for `|Δω| < ε` and slips with period `2π / sqrt(Δω² - ε²)` beyond."""
    trajectory = _run(adler_equation(detuning=detuning), 60.0, 0.05)
    phi = _series(trajectory, 'difference.phi')[0]
    if detuning < 1:
        assert math.sin(phi[-1]) == pytest.approx(-detuning, abs=1e-6)
        return
    slips = np.interp(2 * math.pi * np.arange(1, 8), phi - phi[0], trajectory.times)
    assert np.diff(slips).mean() == pytest.approx(2 * math.pi / math.sqrt(detuning**2 - 1), rel=1e-3)


@pytest.mark.parametrize('shape', ['sinai', 'mushroom'])
def test_billiards_keep_the_speed_and_the_particle_on_the_table(shape):
    """Test that the billiard particle keeps its speed and never leaves the table."""
    trajectory = _run(billiards(shape), 30.0, 0.05, solver=ScipyODE())
    x, y, vx, vy = _series(trajectory, 'particle.x', 'particle.y', 'particle.vx', 'particle.vy')
    assert len(trajectory.events) > 10
    assert np.ptp(vx**2 + vy**2) < 1e-12
    if shape == 'sinai':
        assert np.abs(x).max() <= 1 + 1e-6
        assert np.abs(y).max() <= 1 + 1e-6
        assert np.hypot(x, y).min() >= 0.4 - 1e-6
        return
    stem = y < 0
    assert np.hypot(x, y)[~stem].max() <= 1 + 1e-6
    assert np.abs(x[stem]).max(initial=0.0) <= 0.4 + 1e-6
    assert y.min() >= -1 - 1e-6


@pytest.mark.parametrize(('K', 'synchronized'), [(4.0, True), (0.5, False)])
def test_kuramoto_synchronizes_above_the_critical_coupling(K, synchronized):
    """Test that the Kuramoto order parameter grows above the critical coupling and stays small below it."""
    trajectory = _run(kuramoto(100, K=K), 40.0, 1.0, solver=ScipyODE())
    order = _series(trajectory, 'order')[0]
    assert (order[-10:].mean() > 0.8) == synchronized
    assert (order[-10:].mean() < 0.3) == (not synchronized)


def _random_drift(system):
    """Return the phases at a random state and the drift of a phase model there."""
    problem = compile_system(system)
    state = problem.initial_state.copy()
    state[:] = np.random.default_rng(3).uniform(0, 2 * math.pi, state.size)
    return problem, state, problem.drift(0.0, state, problem.initial_parameters)


def test_kuramoto_in_linear_time():
    """Test that Kuramoto has no space and no pair interaction, and its drift is the all-to-all sum."""
    system = kuramoto(50)
    assert system.space is None
    assert not any(isinstance(interaction, PairInteraction) for interaction in system.interactions)
    problem, theta, rates = _random_drift(system)
    omega = problem.parameters.read(problem.initial_parameters, 'oscillators.omega')
    expected = omega + 4.0 / 50 * np.sin(theta[np.newaxis, :] - theta[:, np.newaxis]).sum(axis=1)
    np.testing.assert_allclose(rates, expected, atol=1e-12)


@pytest.mark.parametrize(('factor', 'synchronized'), [(0.5, False), (2.0, True)])
def test_kuramoto_transition(factor, synchronized):
    """Test that 2000 oscillators stay incoherent at half the critical coupling and synchronize at twice it."""
    critical = 2 * math.sqrt(2 / math.pi)
    trajectory = Simulation(kuramoto(2000, K=factor * critical), dt=0.5).run(until=100).trajectory_
    order = _series(trajectory, 'order')[0][trajectory.times >= 50].mean()
    assert (order > 0.6) == synchronized
    assert (order < 0.1) == (not synchronized)


def test_kuramoto_sakaguchi_populations_form_a_chimera():
    """Test that every population reads both populations and that one synchronizes while the other drifts."""
    system = kuramoto_sakaguchi_chimera()
    assert {(item.target, item.source) for item in system.interactions} == {
        (target, source) for target in ('first', 'second') for source in ('first', 'second')
    }
    trajectory = _run(system, 200.0, 5.0, solver=ScipyODE())
    first, second = _series(trajectory, 'first_order', 'second_order')
    coherent, incoherent = max(first[-1], second[-1]), min(first[-1], second[-1])
    assert coherent > 0.99
    assert 0.3 < incoherent < 0.95


def test_winfree_mean_influence():
    """Test that the Winfree drift is the response to the mean influence, with no pair or self-influence term."""
    system = winfree(50)
    assert len(system.interactions) == 1
    assert not isinstance(system.interactions[0], PairInteraction)
    problem, theta, rates = _random_drift(system)
    omega = problem.parameters.read(problem.initial_parameters, 'oscillators.omega')
    np.testing.assert_allclose(rates, omega - np.sin(theta) * 0.5 / 50 * (1 + np.cos(theta)).sum(), atol=1e-12)


def test_nonlocal_ring_rotates_rigidly_when_synchronized():
    """Test that a synchronized ring turns at `omega - (2R + 1) sin(alpha) / (2R)`, counting every neighbour once."""
    system = nonlocal_ring_chimera(30, R=4, omega=1.0)
    system.elements[0].values['theta'] = np.zeros(30)
    trajectory = _run(system, 2.0, 0.5)
    theta = _series(trajectory, 'ring.theta')[0]
    assert np.allclose(theta[-1], theta[-1][0])
    assert theta[-1][0] == pytest.approx(2.0 * (1.0 - 9 * math.sin(1.46) / 8), rel=1e-6)


@pytest.mark.parametrize(('K', 'expected'), [(2.0, math.sqrt(0.5)), (0.8, 0.0)])
def test_ott_antonsen_order_parameter_reaches_its_fixed_point(K, expected):
    """Test that the order parameter settles at `sqrt(1 - 2 Delta / K)` above `K_c = 2 Delta` and at 0 below."""
    trajectory = _run(ott_antonsen(K=K), 80.0, 1.0)
    x, y = _series(trajectory, 'field.x', 'field.y')
    assert math.hypot(x[-1], y[-1]) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize(('K', 'dead'), [(2.0, True), (0.1, False)])
def test_winfree_oscillators_die_under_strong_coupling(K, dead):
    """Test that strong Winfree coupling stops the oscillators and weak coupling lets them turn."""
    trajectory = _run(winfree(20, K=K), 60.0, 1.0, solver=ScipyODE())
    theta = _series(trajectory, 'oscillators.theta')[0]
    speed = np.abs(theta[-1] - theta[-2]).mean()
    assert (speed < 1e-3) == dead
    assert (speed > 0.5) == (not dead)


def test_mirollo_strogatz_oscillators_fire_in_unison():
    """Test that the integrate-and-fire oscillators, members of one group, end up firing in unison."""
    system = mirollo_strogatz(6)
    assert system.elements[0].size == 6
    trajectory = _run(system, 30.0, 0.1, solver=ScipyODE())
    charges = _series(trajectory, 'oscillators.x')[0]
    assert np.ptp(charges[-1]) < 1e-9
    assert np.ptp(charges[0]) > 0.1
    assert {event.label for event in trajectory.events} == {'fire'}


@pytest.mark.parametrize(('k', 'synchronized'), [(0.5, True), (0.0, False)])
def test_identical_rossler_oscillators_synchronize_with_strong_coupling(k, synchronized):
    """Test that identical Rössler oscillators coupled strongly through their second variable synchronize completely."""
    trajectory = _run(coupled_rossler(k=k, mismatch=0.0), 80.0, 0.5, solver=ScipyODE(rtol=1e-8))
    first, second = _series(trajectory, 'oscillator_0.x1', 'oscillator_1.x1')
    assert (np.abs(first - second)[-20:].max() < 1e-4) == synchronized


@pytest.mark.parametrize(('kappa', 'synchronized'), [(0.45, True), (0.1, False)])
def test_aggregate_averaging_synchronizes_chaotic_maps(kappa, synchronized):
    """Test that logistic maps synchronize above the coupling `3 / 8` and stay apart well below it."""
    trajectory = Simulation(aggregate_averaging_synchronization(kappa=kappa), dt=1).run(until=100).trajectory_
    first, second = _series(trajectory, 'first.x', 'second.x')
    assert (np.abs(first - second)[-20:].max() < 1e-9) == synchronized


def test_plucked_spring_chain_keeps_its_fundamental_mode():
    """Test that the fundamental transverse mode of the chain oscillates at `2 sqrt(k / m) sin(pi / (2 (N + 1)))`."""
    n_masses = 15
    trajectory = Simulation(spring_chain(n_masses), dt=0.1).run(until=4.0).trajectory_
    heights = np.stack([_series(trajectory, f'mass_{index}.x')[0][:, 1] for index in range(n_masses)], axis=1)
    mode = np.sin(np.pi * np.arange(1, n_masses + 1) / (n_masses + 1))
    amplitude = heights @ mode / (mode @ mode)
    frequency = 2 * math.sqrt(1.0 / 0.01) * math.sin(math.pi / (2 * (n_masses + 1)))
    assert amplitude == pytest.approx(amplitude[0] * np.cos(frequency * trajectory.times), abs=2e-4)


def test_standing_wave_returns_after_its_period():
    """Test that a standing wave on a string with fixed ends returns to its shape after `2 L / (n c)`."""
    n_sites, length, mode = 49, 2.0, 2
    spacing = length / (n_sites + 1)
    frequency = 2 / spacing * math.sin(mode * math.pi / (2 * (n_sites + 1)))
    period = 2 * math.pi / frequency
    trajectory = _run(standing_waves(n_sites, length=length, mode=mode), period, period / 20)
    shape = _series(trajectory, 'string.u')[0]
    assert period == pytest.approx(2 * length / mode, rel=1e-3)
    assert np.allclose(shape[-1], shape[0], atol=1e-7)
    assert np.allclose(shape[10], -shape[0], atol=1e-7)


def test_schrodinger_packet_moves_at_the_group_velocity_and_keeps_its_norm():
    """Test that a free wave packet keeps its norm and its center moves at `2 D k`."""
    trajectory = Simulation(schrodinger(), dt=0.5).run(until=4.0).trajectory_
    u, w = _series(trajectory, 'packet.u', 'packet.w')
    density = u**2 + w**2
    positions = 40.0 / 256 * np.arange(256) - 20.0
    center = density @ positions / density.sum(1)
    assert _relative_spread(density.sum(1)) < 1e-6
    assert center[-1] - center[0] == pytest.approx(2 * 0.5 * 2.0 * 4.0, rel=1e-4)


def test_coffee_cools_exponentially_to_the_room():
    """Test that the coffee follows `T_room + (T0 - T_room) exp(-k t)`."""
    trajectory = _run(cooling_coffee(90.0, room=20.0, k=0.1), 30.0, 1.0)
    expected = 20.0 + 70.0 * np.exp(-0.1 * trajectory.times)
    assert _series(trajectory, 'coffee.T')[0] == pytest.approx(expected, rel=1e-8)


def test_einstein_de_sitter_universe_grows_as_two_thirds_power():
    """Test that a flat universe of matter expands as `(1 + 3 H0 t / 2)^(2/3)`."""
    trajectory = _run(friedmann_expansion(omega_matter=1.0, omega_lambda=0.0), 3.0, 0.1)
    expected = (1 + 1.5 * trajectory.times) ** (2 / 3)
    assert _series(trajectory, 'universe.a')[0] == pytest.approx(expected, rel=1e-8)


def test_dark_energy_universe_keeps_the_friedmann_equation_and_accelerates():
    """Test that the universe with dark energy keeps the Friedmann equation as a first integral and accelerates."""
    trajectory = _run(friedmann_expansion(), 2.0, 0.05)
    a, rate = _series(trajectory, 'universe.a', 'universe.a_d1')
    friedmann = rate**2 - (0.3 / a + 0.7 * a**2)
    assert np.abs(friedmann).max() < 1e-8
    assert np.all(np.diff(rate) > 0)


def test_sitnikov_conserves_energy_with_circular_primaries_and_stays_symmetric():
    """Test that the Sitnikov body conserves its energy above circular primaries and oscillates symmetrically."""
    trajectory = _run(sitnikov(0.5), 30.0, 0.05)
    z, speed = _series(trajectory, 'body.z', 'body.z_d1')
    energy = speed**2 / 2 - 1 / np.sqrt(z**2 + 0.25)
    assert np.abs(energy - energy[0]).max() < 1e-6
    assert z.min() == pytest.approx(-0.5, abs=1e-4)
