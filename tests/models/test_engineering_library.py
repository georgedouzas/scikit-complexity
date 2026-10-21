"""Test the engineering models: traffic, circuits, control, power systems, robotics, and processes."""

import math

import numpy as np
import pytest
from sage.all__sagemath_symbolics import sqrt, tanh

from skcomplex.core.base import ParameterError
from skcomplex.core.compile import compile_system
from skcomplex.core.model import Element, Interaction, Rate, System
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4, SSPRK, EulerMaruyama, ScipyODE
from skcomplex.models.engineering.circuits import (
    chua,
    impulsive_circuits,
    memristor_oscillator,
    mixed_mode_chua,
    tunnel_diode_oscillator,
    vibration_energy_harvester,
)
from skcomplex.models.engineering.control import pendulum_swing_up, pid_loop, satellite_transfer, thermostat
from skcomplex.models.engineering.power_systems import (
    faraday_disk,
    power_grid,
    regenerative_braking,
    wheel_rail_contact,
)
from skcomplex.models.engineering.processes import bathtub, cascade_classifier
from skcomplex.models.engineering.robotics import light_seeking_robots, swarmtaxis
from skcomplex.models.engineering.traffic import (
    charbonneau_traffic,
    delayed_car_following,
    intelligent_driver_model,
    kerner_konhauser,
    lighthill_whitham_richards,
    optimal_velocity,
    payne,
    ring_road_drivers,
)


def _cars(trajectory, n_cars, variable):
    """Return a variable of every car as an array of shape (frames, cars)."""
    return np.array([trajectory[f'car_{index}.{variable}'] for index in range(n_cars)]).T


Car = Element.define('Car', states={'x': 0.0, 'v': 0.0}, laws=lambda s: [Rate(s.x, s.v)])


def _peaks(series):
    """Return the values of the local maxima of a series."""
    inner = series[1:-1]
    return inner[(inner > series[:-2]) & (inner > series[2:])]


def test_optimal_velocity_jams_move_backwards():
    """Test that uniform flow above the critical density breaks into a stop-and-go wave that moves backwards."""
    n_cars, length = 12, 24.0
    trajectory = Simulation(optimal_velocity(n_cars, length), dt=1.0).run(until=150).trajectory_
    positions, speeds = trajectory['cars.x'], trajectory['cars.v']
    assert np.ptp(speeds[0]) < 0.01
    assert speeds[-1].min() < 0.2
    assert speeds[-1].max() > 1.5
    jam = np.array([positions[frame, np.argmin(speeds[frame])] % length for frame in range(100, 151)])
    shifts = (np.diff(jam) + length / 2) % length - length / 2
    assert shifts.sum() < -5.0
    assert speeds[100:].mean() > 0.5


def test_optimal_velocity_below_critical_density_stays_uniform():
    """Test that a perturbed uniform flow with large headways relaxes back to uniform flow."""
    n_cars, length = 8, 32.0
    trajectory = Simulation(optimal_velocity(n_cars, length), dt=1.0).run(until=80).trajectory_
    speeds = trajectory['cars.v']
    assert np.ptp(speeds[-1]) < 0.01


def test_intelligent_drivers_reach_equilibrium_speed():
    """Test that cars at low density reach the equilibrium speed of their gap, `s = s*(v) / sqrt(1 - (v / v0)^4)`."""
    trajectory = Simulation(intelligent_driver_model(10, 250.0), dt=1.0).run(until=100).trajectory_
    speed = trajectory['cars.v'][-1].mean()
    gap = 250.0 / 10 - 5.0
    assert gap == pytest.approx((2.0 + speed * 1.5) / math.sqrt(1 - (speed / 30.0) ** 4), rel=1e-3)


PerCarVelocity = Interaction.define(
    'PerCarVelocity',
    roles=['leader', 'follower'],
    parameters={'a': 1.0, 'V0': 1.0, 'hc': 2.0, 'offset': 0.0},
    laws=lambda s: [
        Rate(
            s.follower.v,
            s.a * (s.V0 * (tanh(s.leader.x - s.follower.x + s.offset - s.hc) + tanh(s.hc)) - s.follower.v),
        ),
    ],
)

PerCarDriver = Interaction.define(
    'PerCarDriver',
    roles=['leader', 'follower'],
    parameters={'v0': 30.0, 'T': 1.5, 'a': 1.0, 'b': 1.5, 's0': 2.0, 'delta': 4.0, 'length': 5.0, 'offset': 0.0},
    laws=lambda s: [
        Rate(
            s.follower.v,
            s.a
            * (
                1
                - (s.follower.v / s.v0) ** s.delta
                - (
                    (s.s0 + s.follower.v * s.T + s.follower.v * (s.follower.v - s.leader.v) / (2 * sqrt(s.a * s.b)))
                    / (s.leader.x - s.follower.x + s.offset - s.length)
                )
                ** 2
            ),
        ),
    ],
)


def _per_car(system, interaction):
    """Return the ring road of a group of cars rebuilt as one element and one interaction per car, as before."""
    cars, following = system.elements[0], system.interactions[0]
    n_cars, road = cars.size, following.road_length
    elements = [
        Car(f'car_{index}', x=cars.values['x'][index], v=float(np.broadcast_to(cars.values['v'], n_cars)[index]))
        for index in range(n_cars)
    ]
    settings = {
        key: value for key, value in following.get_params().items() if key not in {'label', 'cars', 'road_length'}
    }
    interactions = [
        interaction(
            f'follow_{index}',
            leader=f'car_{(index + 1) % n_cars}',
            follower=f'car_{index}',
            offset=road if index == n_cars - 1 else 0.0,
            **settings,
        )
        for index in range(n_cars)
    ]
    return System(elements=elements, interactions=interactions)


def test_ring_roads_hold_one_group_of_cars():
    """Test that a ring road holds one group and one interaction whatever the number of cars."""
    system = optimal_velocity(n_cars=50)
    assert [element.size for element in system.elements] == [50]
    assert len(system.interactions) == 1
    trajectory = Simulation(system, dt=1.0).run(until=2).trajectory_
    assert trajectory['cars.x'].shape == trajectory['cars.v'].shape == (3, 50)


@pytest.mark.parametrize(
    ('factory', 'interaction', 'until'),
    [(optimal_velocity, PerCarVelocity, 200.0), (intelligent_driver_model, PerCarDriver, 100.0)],
)
def test_ring_roads_match_one_interaction_per_car(factory, interaction, until):
    """Test that the ring road of one group runs as the ring of one element and one interaction per car under RK45."""
    system = factory(n_cars=20)
    group = Simulation(system, dt=1.0, solver=ScipyODE(method='RK45')).run(until=until).trajectory_
    per_car = Simulation(_per_car(system, interaction), dt=1.0, solver=ScipyODE(method='RK45')).run(until=until)
    per_car = per_car.trajectory_
    for variable in ('x', 'v'):
        np.testing.assert_allclose(group[f'cars.{variable}'], _cars(per_car, 20, variable), rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize(('reaction', 'grows'), [(1.0, True), (0.3, False)])
def test_delayed_car_following_instability(reaction, grows):
    """Test that oscillations grow along the chain when `omega < 2 reaction sin(omega T)` and decay otherwise."""
    trajectory = Simulation(delayed_car_following(5, reaction=reaction), dt=0.5).run(until=60).trajectory_
    amplitudes = np.ptp(_cars(trajectory, 5, 'v')[60:], axis=0)
    assert (amplitudes[-1] > amplitudes[0]) == grows


def test_lighthill_whitham_richards_shock_speed():
    """Test that the shock between free and queued traffic moves at the Rankine-Hugoniot speed."""
    trajectory = Simulation(lighthill_whitham_richards(), dt=0.05, solver=SSPRK()).run(until=1.0).trajectory_
    density = trajectory['road.rho'][-1]
    centers = (np.arange(200) + 0.5) / 200
    assert centers[np.argmax(np.diff(density))] == pytest.approx(0.5 + (1 - 1.1), abs=0.01)


def test_payne_conserves_cars_and_keeps_equilibrium():
    """Test that the Payne model conserves the cars and keeps uniform flow at the equilibrium speed."""
    trajectory = Simulation(payne(50), dt=0.1, solver=SSPRK()).run(until=0.5).trajectory_
    assert trajectory['road.rho'].sum(axis=1) == pytest.approx(np.full(6, trajectory['road.rho'][0].sum()))
    uniform = Simulation(payne(50, bump=0.0), dt=0.1, solver=SSPRK()).run(until=0.5).trajectory_
    assert np.ptp(uniform['road.q'][-1]) < 1e-12


def test_kerner_konhauser_forms_a_cluster():
    """Test that a small bump of density grows into a dense cluster while the cars are conserved."""
    trajectory = Simulation(kerner_konhauser(), dt=10.0, solver=SSPRK(cfl=0.15)).run(until=40).trajectory_
    density = trajectory['road.rho']
    assert density[-1].max() > 0.35
    assert density.sum(axis=1)[-1] == pytest.approx(density.sum(axis=1)[0])


def test_charbonneau_cars_keep_order_and_distance():
    """Test that cars never pass each other or come closer than one unit, and that random braking stops some cars."""
    trajectory = Simulation(charbonneau_traffic(50), random_state=0).run(until=100).trajectory_
    snapshot = trajectory.agents('cars', 100)
    order = np.argsort(snapshot.values['rank'])
    assert np.diff(snapshot.values['x'][order]).min() >= 1.0 - 1e-9
    assert snapshot.values['v'].min() < snapshot.values['v'].max()


@pytest.mark.parametrize(('n_cars', 'eps', 'free'), [(20, 0.0, True), (30, 0.02, False)])
def test_ring_road_drivers_jam_with_errors(n_cars, eps, free):
    """Test that sparse drivers without errors reach the speed limit, and errors of speed jam a denser road."""
    trajectory = Simulation(ring_road_drivers(n_cars, eps=eps), random_state=0).run(until=100).trajectory_
    speeds = trajectory.agents('drivers', 100).values['speed']
    assert (speeds.min() == pytest.approx(40.0)) == free


def test_ring_road_drivers_read_the_road_length_from_the_space():
    """Test that drivers keep no road length of their own and stay on the ring of the space's width."""
    system = ring_road_drivers(10, length=500.0)
    assert 'road_length' not in system.elements[0].element.__dataclass_fields__
    trajectory = Simulation(system, random_state=0).run(until=100).trajectory_
    positions = np.concatenate([trajectory.agents('drivers', frame).locations[:, 0] for frame in range(101)])
    assert positions.min() >= 0.0
    assert positions.max() < 500.0


def test_chua_double_scroll_visits_both_scrolls():
    """Test that the double-scroll attractor stays bounded and visits both sides of the origin."""
    trajectory = Simulation(chua(), dt=0.05).run(until=100).trajectory_
    x = trajectory['circuit.x'][400:]
    assert np.abs(x).max() < 3.0
    assert 0.2 < (x > 0).mean() < 0.8


def test_chua_sine_diode_has_several_scrolls():
    """Test that the sine diode gives a bounded attractor spanning more than two scrolls."""
    trajectory = Simulation(chua(nonlinearity='sine'), dt=0.5).run(until=600).trajectory_
    x = trajectory['circuit.x'][100:]
    assert np.abs(x).max() < 15.0
    assert x.min() < -4 * 1.3
    assert x.max() > 4 * 1.3


def test_chua_rejects_unknown_nonlinearity():
    """Test that an unknown diode characteristic raises a parameter error."""
    with pytest.raises(ParameterError, match='Unknown nonlinearity'):
        chua(nonlinearity='cubic')


@pytest.mark.parametrize(('c', 'mixed'), [(0.00053, True), (0.00105, False)])
def test_mixed_mode_chua_oscillations(c, mixed):
    """Test that small oscillations come between the spikes for `c = 0.00053` and not for `c = 0.00105`."""
    solver = ScipyODE(method='Radau')
    trajectory = Simulation(mixed_mode_chua(c=c), dt=0.1, solver=solver).run(until=50).trajectory_
    peaks = _peaks(trajectory['circuit.x'][100:])
    assert (peaks > 1.0).sum() >= 4
    assert ((peaks < 0.3).sum() >= 4) == mixed


def test_memristor_oscillator_keeps_oscillating():
    """Test that the memristor circuit neither decays nor diverges."""
    trajectory = Simulation(memristor_oscillator(), dt=0.1).run(until=200).trajectory_
    voltage = trajectory['circuit.v']
    assert np.abs(voltage).max() < 10.0
    assert np.ptp(voltage[-500:]) > 0.5


def test_tunnel_diode_limit_cycle_amplitude():
    """Test that the tunnel-diode oscillator settles on the van der Pol limit cycle of amplitude `2 sqrt(alpha / 3)`."""
    trajectory = Simulation(tunnel_diode_oscillator(alpha=1.0, gamma=1.0), dt=0.05).run(until=50).trajectory_
    assert np.abs(trajectory['circuit.V'][-300:]).max() == pytest.approx(2 * math.sqrt(1 / 3), rel=0.02)


def test_impulsive_circuit_kicks_the_rate():
    """Test that every impulse fires once per period and raises the rate by `A / (L C)`."""
    system = impulsive_circuits(A=0.5, L=2.0, C=0.5, T=2.0)
    rate = compile_system(system).states.names.index('circuit.x_d1')
    trajectory = Simulation(system, dt=0.05).run(until=10).trajectory_
    events = [event for event in trajectory.events if event.label == 'impulse']
    assert [round(event.time, 6) for event in events] == [2.0, 4.0, 6.0, 8.0, 10.0][: len(events)]
    assert len(events) >= 4
    assert all(event.after[rate] - event.before[rate] == pytest.approx(0.5) for event in events)


def test_vibration_energy_harvester_steady_response():
    """Test that the sprung mass settles on the amplitude of the linear response, and the voltage follows the stroke."""
    trajectory = Simulation(vibration_energy_harvester(), dt=0.01).run(until=30).trajectory_
    m1, m2, k1, k2, b, A, omega = 40.0, 400.0, 2.0e5, 2.0e4, 1500.0, 0.01, 9.42
    stiffness = np.array(
        [
            [k1 + k2 - m1 * omega**2 + 1j * omega * b, -k2 - 1j * omega * b],
            [-k2 - 1j * omega * b, k2 - m2 * omega**2 + 1j * omega * b],
        ],
    )
    amplitude = abs(np.linalg.solve(stiffness, [k1 * A, 0.0])[1])
    sprung = trajectory['harvester.y2'][-1000:]
    assert np.ptp(sprung) / 2 == pytest.approx(amplitude, rel=0.01)
    stroke = trajectory['harvester.y2'] - trajectory['harvester.y1']
    assert trajectory['voltage'] == pytest.approx(10.0 * 4.0e-10 * 2.0e4 * stroke / 1.0e-8)


def test_thermostat_settles_below_the_setting():
    """Test that the leaky room settles at `(setting + leak * outside) / (1 + leak)`."""
    trajectory = Simulation(thermostat(), dt=0.25).run(until=48).trajectory_
    assert trajectory['room.T'][-1] == pytest.approx((18 + 0.1 * 10) / 1.1, rel=1e-4)


@pytest.mark.parametrize(('Ki', 'final'), [(1.0, 1.0), (0.0, 0.5)])
def test_pid_loop_integral_action(Ki, final):
    """Test that integral action removes the error left by a disturbance, which proportional control keeps."""
    trajectory = Simulation(pid_loop(Ki=Ki), dt=0.1).run(until=40).trajectory_
    assert trajectory['process.y'][-1] == pytest.approx(final, abs=1e-3)


def test_pendulum_swing_up_reaches_the_top():
    """Test that the energy control brings the pendulum to its inverted position."""
    trajectory = Simulation(pendulum_swing_up(), dt=0.1, solver=RK4(dt=0.01)).run(until=100).trajectory_
    theta, omega = trajectory['pendulum.theta'][-100:], trajectory['pendulum.theta_d1'][-100:]
    assert np.cos(theta) == pytest.approx(np.full(100, -1.0), abs=0.01)
    assert np.abs(omega[-10:]).max() < 0.01


def test_satellite_transfer_time():
    """Test that the transfer from the periapsis reaches the final energy after about 9.81."""
    trajectory = Simulation(satellite_transfer(), dt=0.05, solver=RK4(dt=0.005)).run(until=15).trajectory_
    energy = trajectory['energy']
    assert energy[0] == pytest.approx(-0.7, abs=1e-3)
    assert trajectory.times[np.argmax(energy >= -0.2)] == pytest.approx(9.81, abs=0.1)
    assert energy[-1] == pytest.approx(-0.2, abs=1e-3)


@pytest.mark.parametrize(('changes', 'synchronous'), [({}, True), ({'added_line': (0, 2)}, False)])
def test_power_grid_braess_paradox(changes, synchronous):
    """Test that the ring grid synchronises, and loses synchrony when a line is added."""
    trajectory = Simulation(power_grid(**changes), dt=1.0).run(until=100).trajectory_
    frequencies = np.array([trajectory[f'node_{index}.phi_d1'][-10:] for index in range(8)])
    assert (np.abs(frequencies).max() < 0.1) == synchronous


def test_regenerative_braking_stores_energy():
    """Test that braking charges the capacitor, slows the car, and never creates energy."""
    trajectory = Simulation(regenerative_braking(), dt=0.5).run(until=30).trajectory_
    speed, current, voltage = trajectory['car.v'], trajectory['car.I'], trajectory['car.Vs']
    energy = 1000.0 * speed**2 / 2 + 680.0 * voltage**2 / 2 + 0.01 * current**2 / 2
    assert np.all(np.diff(energy) < 1e-6)
    assert speed[-1] < speed[0] / 2
    assert voltage[-1] > 5.0


def test_faraday_disk_reaches_torque_balance():
    """Test that the charger settles where the drive torque balances friction and the braking Lorentz torque."""
    trajectory = Simulation(faraday_disk(), dt=0.1).run(until=50).trajectory_
    current, voltage, omega = (trajectory[f'charger.{name}'][-1] for name in ('I', 'U', 'omega'))
    coupling = (1.5 + 1.2566370614359173e-4 * 250 * current / 0.05) * (0.55**2 - 0.025**2) / 2
    assert 0.5 + 0.5 * 0.55 * omega + current * coupling == pytest.approx(4.8, abs=1e-4)
    assert voltage == pytest.approx(109.0 * current, abs=1e-4)


def test_faraday_disk_charges_only_above_the_battery_voltage():
    """Test that the default disk discharges the battery steadily and a fast disk makes the current oscillate."""
    slow = Simulation(faraday_disk(), dt=0.1).run(until=50).trajectory_
    fast = Simulation(faraday_disk(M=11.0), dt=0.1).run(until=200).trajectory_
    current, omega = slow['charger.I'][-1], slow['charger.omega'][-1]
    assert current == pytest.approx(-0.040, abs=1e-3)
    assert 1.5 * (0.55**2 - 0.025**2) / 2 * omega < 8.0
    assert np.ptp(fast['charger.I'][-500:]) > 1.0


def test_wheel_rail_response_is_periodic():
    """Test that the rail settles on a response with the period of the passing bogies."""
    trajectory = Simulation(wheel_rail_contact(), dt=0.05).run(until=60).trajectory_
    rail = trajectory['rail.z']
    assert rail[-1] == pytest.approx(rail[-201], abs=1e-4)
    assert rail.max() > 0.1


def test_light_seeking_robots_avoid_light_with_delay():
    """Test that robots with a positive sensory delay spend less time in the light than robots without one."""
    intensities = []
    for delay in (0.0, 5.0):
        system = light_seeking_robots(40, delay=delay)
        trajectory = Simulation(system, dt=0.5, solver=EulerMaruyama(dt=0.01), random_state=1).run(until=150)
        x, y = trajectory.trajectory_['robots.x'], trajectory.trajectory_['robots.y']
        intensities.append(np.exp(-(x**2 + y**2))[trajectory.trajectory_.times > 50].mean())
    assert intensities[1] < 0.9 * intensities[0]


def test_light_seeking_robots_stay_in_the_well():
    """Test that the wall keeps every robot inside the well."""
    trajectory = Simulation(light_seeking_robots(10), dt=1.0, random_state=0).run(until=20).trajectory_
    assert np.hypot(trajectory['robots.x'], trajectory['robots.y']).max() <= 2.0 + 1e-9


def test_swarmtaxis_moves_towards_the_beacon():
    """Test that the centre of the swarm moves towards the beacon and the swarm stays together."""
    trajectory = Simulation(swarmtaxis(), random_state=0).run(until=300).trajectory_
    start = np.asarray(trajectory.agents('robots', 0).locations).mean(axis=0)
    end = np.asarray(trajectory.agents('robots', 300).locations)
    assert end.mean(axis=0)[0] - start[0] > 1.0
    assert end.std(axis=0).max() < 10.0


def test_bathtub_drains_then_holds():
    """Test that the tub loses five gallons per minute until the faucet opens, and then holds its level."""
    trajectory = Simulation(bathtub(), dt=0.5).run(until=10).trajectory_
    assert trajectory['tub.water'][10] == pytest.approx(25.0)
    assert trajectory['tub.water'][-1] == pytest.approx(25.0)


@pytest.mark.parametrize(('n_stages', 'k', 'feed_stage'), [(10, 0.6, None), (10, 0.45, 3)])
def test_cascade_classifier_fractional_extraction(n_stages, k, feed_stage):
    """Test that the share of the feed in the fine product matches Barsky's fractional extraction."""
    system = cascade_classifier(n_stages, k=k, feed_stage=feed_stage)
    trajectory = Simulation(system, dt=10.0).run(until=400).trajectory_
    fine = trajectory['fine.m']
    entry = (n_stages + 1) // 2 if feed_stage is None else feed_stage
    ratio = (1 - k) / k
    expected = (1 - ratio ** (n_stages + 1 - entry)) / (1 - ratio ** (n_stages + 1))
    assert (fine[-1] - fine[-11]) / 100 == pytest.approx(expected, rel=1e-4)
