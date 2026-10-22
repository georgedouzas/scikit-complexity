"""Test the statistical mechanics, criticality, soft matter, active matter, and fluids models of physics."""

import numpy as np
import pytest
from scipy import ndimage

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import SSPRK, EulerMaruyama
from skcomplex.models.physics.active_matter import (
    active_brownian_particles,
    active_ornstein_uhlenbeck_particles,
    chiral_active_particles,
    cucker_smale_flocking,
    depot_brownian_particles,
    living_crystals,
    motility_induced_phase_separation,
    run_and_tumble,
    toner_tu,
)
from skcomplex.models.physics.criticality import manna_sandpile
from skcomplex.models.physics.fluids import discrete_velocity_boltzmann, wave_breaking
from skcomplex.models.physics.soft_matter import (
    brownian_motion,
    diffusion_in_a_speckle_field,
    kramers_escape,
    multiplicative_noise,
    optical_trap,
    scaled_brownian_motion,
)
from skcomplex.models.physics.statistical_mechanics import (
    contact_process,
    diffusion_limited_aggregation,
    discrete_master_equation,
    landau_relaxation,
    q2r,
    site_percolation,
    tasep,
    two_phase_zone_exchange,
)


def _long_time_slope(times, displacements, start):
    """Return the slope of the mean squared displacement against time after a start index."""
    squared = ((displacements - displacements[0]) ** 2).sum(axis=-1).mean(axis=-1)
    return np.polyfit(times[start:], squared[start:], 1)[0]


def _mean_neighbors(positions, size, radius):
    """Return the mean number of neighbours closer than a radius in a periodic box."""
    separations = positions[:, None] - positions[None]
    separations -= size * np.rint(separations / size)
    distances = np.linalg.norm(separations, axis=-1)
    np.fill_diagonal(distances, np.inf)
    return (distances < radius).sum(axis=1).mean()


def _ising_energy(spins):
    """Return the nearest-neighbour Ising energy of a periodic square lattice."""
    return -(spins * np.roll(spins, 1, axis=0)).sum() - (spins * np.roll(spins, 1, axis=1)).sum()


def test_active_brownian_particles_diffuse_with_the_active_coefficient():
    """Test that active Brownian particles spread as `(4 D_t + 2 v^2 / D_r) t` at long times within 5%."""
    speed, D_r, D_t = 3.0, 1.0, 0.1
    model = active_brownian_particles(4000, speed=speed, D_r=D_r, D_t=D_t)
    trajectory = Simulation(model, dt=1.0, solver=EulerMaruyama(dt=0.01), random_state=0).run(until=50).trajectory_
    slope = _long_time_slope(trajectory.times, trajectory['swimmers.x'], 10)
    assert slope == pytest.approx(4 * D_t + 2 * speed**2 / D_r, rel=0.05)


def test_chiral_active_particles_diffuse_slower_when_they_turn():
    """Test that chiral particles diffuse with `D_t + v^2 D_r / (2 (D_r^2 + omega^2))`."""
    model = chiral_active_particles(1000, speed=3.0, omega=2.0, D_t=0.1, D_r=0.1)
    trajectory = Simulation(model, dt=1.0, solver=EulerMaruyama(dt=0.01), random_state=0).run(until=40).trajectory_
    slope = _long_time_slope(trajectory.times, trajectory['swimmers.x'], 10)
    assert slope / 4 == pytest.approx(0.1 + 9 * 0.1 / (2 * (0.01 + 4)), rel=0.1)


def test_active_ornstein_uhlenbeck_particles_diffuse_with_the_active_coefficient():
    """Test that Ornstein-Uhlenbeck propulsion gives the diffusion coefficient `D` at long times."""
    model = active_ornstein_uhlenbeck_particles(1000, tau=1.0, D=1.0)
    trajectory = Simulation(model, dt=1.0, solver=EulerMaruyama(dt=0.01), random_state=0).run(until=40).trajectory_
    assert _long_time_slope(trajectory.times, trajectory['swimmers.x'], 10) / 4 == pytest.approx(1.0, rel=0.1)


def test_run_and_tumble_diffuses_with_the_persistence_of_runs():
    """Test that run-and-tumble agents diffuse with `v^2 (1 + q) / (4 (1 - q))` per step."""
    trajectory = Simulation(run_and_tumble(200), random_state=0).run(until=200).trajectory_
    positions = np.array([trajectory.agents('bacteria', frame).locations for frame in range(len(trajectory.times))])
    steps = np.diff(positions.astype(float), axis=0)
    steps -= 10.0 * np.rint(steps / 10.0)
    paths = np.concatenate([np.zeros((1, *steps.shape[1:])), np.cumsum(steps, axis=0)])
    persistence = np.exp(-0.1)
    expected = 0.01 * (1 + persistence) / (4 * (1 - persistence))
    assert _long_time_slope(trajectory.times, paths, 50) / 4 == pytest.approx(expected, rel=0.15)


def test_depot_particles_reach_the_pumped_speed():
    """Test that depot-pumped particles start at rest and move at `sqrt(q / gamma - c / d)`."""
    trajectory = Simulation(depot_brownian_particles(), dt=0.5, random_state=0).run(until=10).trajectory_
    speeds = np.linalg.norm(trajectory['walkers.v'][-1], axis=-1)
    assert speeds.mean() == pytest.approx(np.sqrt(10 / 20 - 1 / 10), rel=0.05)


def test_motility_induced_phase_separation_gathers_disks():
    """Test that fast repulsive swimmers gather into clusters without passing through each other."""
    model = motility_induced_phase_separation(200)
    size = model.space.width
    trajectory = Simulation(model, dt=0.5, solver=EulerMaruyama(dt=0.005), random_state=0).run(until=5).trajectory_
    positions = trajectory['swimmers.x']
    assert _mean_neighbors(positions[-1], size, 1.1) > 3 * _mean_neighbors(positions[0], size, 1.1)
    distances = np.linalg.norm(positions[-1][:, None] - positions[-1][None], axis=-1)
    assert distances[np.triu_indices(len(distances), 1)].min() > 0.8


def test_living_crystals_form_only_with_phoretic_attraction():
    """Test that active colloids gather into clusters with phoretic attraction and stay apart without it."""
    neighbors = {}
    for v0 in (0.0, 50.0):
        model = living_crystals(50, size=30.0, v0=v0)
        solver = EulerMaruyama(dt=0.002)
        trajectory = Simulation(model, dt=0.5, solver=solver, random_state=0).run(until=5).trajectory_
        neighbors[v0] = _mean_neighbors(trajectory['swimmers.x'][-1], 30.0, 2.5)
    assert neighbors[50.0] > 4 * neighbors[0.0]


def test_cucker_smale_flock_aligns_and_keeps_momentum():
    """Test that Cucker-Smale velocities converge to their conserved mean."""
    trajectory = Simulation(cucker_smale_flocking(), dt=1.0, random_state=0).run(until=20).trajectory_
    velocities = trajectory['birds.v']
    assert np.allclose(velocities[-1].mean(axis=0), velocities[0].mean(axis=0), atol=1e-6)
    assert velocities[-1].std(axis=0).max() < 1e-2 * velocities[0].std(axis=0).min()


def test_toner_tu_flock_orders_at_the_mean_field_speed():
    """Test that the polar fluid conserves its mass and moves together at `sqrt(a (rho0 - rho_c) / beta)`."""
    model = toner_tu((16, 16))
    trajectory = Simulation(model, dt=1.0, solver=EulerMaruyama(dt=0.02), random_state=0).run(until=30).trajectory_
    vx, vy, rho = trajectory['flock.vx'][-1], trajectory['flock.vy'][-1], trajectory['flock.rho']
    assert rho[-1].sum() == pytest.approx(rho[0].sum())
    assert np.hypot(vx.mean(), vy.mean()) == pytest.approx(np.sqrt(0.5), rel=0.02)


def test_brownian_motion_is_ballistic_then_diffusive():
    """Test that inertial Brownian motion is ballistic below `m / gamma` and diffusive above, and overdamped is not."""
    model = brownian_motion(4000)
    trajectory = Simulation(model, dt=0.05, solver=EulerMaruyama(dt=0.001), random_state=0).run(until=1).trajectory_
    positions = trajectory['beads.x'][..., 0]
    assert (positions[2] ** 2).mean() == pytest.approx(2 * (0.1 - 1 + np.exp(-0.1)), rel=0.1)
    overdamped = brownian_motion(4000, inertial=False)
    trajectory = Simulation(overdamped, dt=0.5, solver=EulerMaruyama(dt=0.01), random_state=0).run(until=10).trajectory_
    assert (trajectory['beads.x'][-1] ** 2).mean() == pytest.approx(2 * 10, rel=0.1)


def test_optical_trap_holds_particles_with_the_thermal_variance():
    """Test that trapped particles have the variance `kT / k` along each axis, and that moving traps run."""
    trajectory = Simulation(optical_trap(1000), dt=1, solver=EulerMaruyama(dt=0.01), random_state=0).run(until=30)
    positions = trajectory.trajectory_['beads.x'][10:]
    assert positions[..., 0].var() == pytest.approx(1.0, rel=0.1)
    assert positions[..., 1].var() == pytest.approx(4.0, rel=0.1)
    blinking = optical_trap(100, blink_period=4.0, amplitude=(1.0, 0.0), frequency=0.1)
    moving = Simulation(blinking, dt=1, solver=EulerMaruyama(dt=0.01), random_state=0).run(until=10).trajectory_
    assert np.isfinite(moving['beads.x']).all()


def test_kramers_escape_rate():
    """Test that particles hop between the wells at the Kramers rate within 25%."""
    D, size, until = 0.1, 200, 400
    solver = EulerMaruyama(dt=0.01)
    trajectory = (
        Simulation(kramers_escape(size, D=D), dt=0.1, solver=solver, random_state=0).run(until=until).trajectory_
    )
    positions = trajectory['particles.x']
    wells = np.sign(positions[0])
    hops = 0
    for row in positions[1:]:
        following = np.where(row > 0.5, 1, np.where(row < -0.5, -1, wells))
        hops += (following != wells).sum()
        wells = following
    expected = np.sqrt(2) / (2 * np.pi) * np.exp(-1 / (4 * D))
    assert hops / (size * until) == pytest.approx(expected, rel=0.25)


@pytest.mark.parametrize(('noise', 'left'), [('ito', 0.75), ('stratonovich', np.log(2) / np.log(3)), ('anti-ito', 0.5)])
def test_multiplicative_noise_conventions_give_their_stationary_densities(noise, left):
    """Test that the fraction in the low-noise half follows the stationary density of each convention."""
    model = multiplicative_noise(500, s0=1.0, ds=1.0, L=10.0, convention=noise)
    trajectory = Simulation(model, dt=0.05, solver=EulerMaruyama(dt=0.01), random_state=0).run(until=400).trajectory_
    positions = trajectory['particles.x'][2000:]
    assert (positions < 5).mean() == pytest.approx(left, abs=0.03)


def test_scaled_brownian_motion_has_the_anomalous_exponent():
    """Test that the mean squared displacement of scaled Brownian motion grows as `t^alpha`."""
    model = scaled_brownian_motion(4000, alpha=0.5)
    trajectory = Simulation(model, dt=0.1, solver=EulerMaruyama(dt=0.001), random_state=0).run(until=10).trajectory_
    squared = (trajectory['walkers.x'] ** 2).mean(axis=1)
    exponent = np.polyfit(np.log(trajectory.times[10:]), np.log(squared[10:]), 1)[0]
    assert exponent == pytest.approx(0.5, abs=0.05)


def test_speckle_field_traps_particles():
    """Test that particles diffuse freely without the speckle field and are trapped by a strong one."""
    coefficients = {}
    for strength in (0.0, 1000.0):
        model = diffusion_in_a_speckle_field(200, strength=strength)
        solver = EulerMaruyama(dt=0.01)
        trajectory = Simulation(model, dt=1, solver=solver, random_state=0).run(until=50).trajectory_
        positions = trajectory['beads.x']
        coefficients[strength] = ((positions[-1] - positions[20]) ** 2).sum(axis=-1).mean() / (4 * 30)
    assert coefficients[0.0] == pytest.approx(1.0, rel=0.15)
    assert coefficients[1000.0] < 0.1


def test_speckle_field_defaults_give_anomalous_diffusion():
    """Test that the default field draws the particles in over several frames and lets them escape slowly."""
    solver = EulerMaruyama(dt=0.01)
    simulation = Simulation(diffusion_in_a_speckle_field(), dt=0.5, solver=solver, random_state=0).run(until=60)
    positions = np.asarray(simulation.trajectory_['beads.x'])
    times = simulation.trajectory_.times[21:] - simulation.trajectory_.times[20]
    squared = ((positions[21:] - positions[20]) ** 2).sum(axis=-1).mean(axis=-1)
    exponent = np.polyfit(np.log(times), np.log(squared), 1)[0]
    assert np.linalg.norm(positions[1] - positions[0], axis=-1).mean() < 5
    assert 0.1 < exponent < 0.8


def test_q2r_conserves_energy():
    """Test that the Q2R automaton keeps the Ising energy while the magnetisation changes."""
    trajectory = Simulation(q2r((32, 32)), random_state=0).run(until=50).trajectory_
    spins = trajectory.lattice('spins.spin')
    assert {_ising_energy(frame) for frame in spins} == {_ising_energy(spins[0])}
    assert abs(spins[-1].mean()) < abs(spins[0].mean())


@pytest.mark.parametrize(('p', 'spans'), [(0.45, False), (0.75, True)])
def test_site_percolation_burns_the_clusters_of_the_top_row(p, spans):
    """Test that the fire burns the clusters touching the top row, which span the lattice only above threshold."""
    model = site_percolation((40, 40), p=p, random_state=1)
    occupied = np.reshape(model.elements[0].values['cover'], (40, 40)) > 0
    trajectory = Simulation(model, random_state=0).run(until=200).trajectory_
    burnt = trajectory.lattice('lattice.cover')[-1] == 3
    labels, _ = ndimage.label(occupied)
    lit = np.isin(labels, np.setdiff1d(labels[0], [0]))
    assert np.array_equal(burnt, lit)
    assert burnt[-1].any() == spans


@pytest.mark.parametrize(('rate', 'survives'), [(0.5, False), (4.0, True)])
def test_contact_process_has_an_absorbing_transition(rate, survives):
    """Test that the infection dies out below the critical rate and persists above it."""
    trajectory = Simulation(contact_process((32, 32), rate=rate), random_state=0).run(until=300).trajectory_
    assert (trajectory.lattice('lattice.infected')[-1].mean() > 0.3) == survives


def test_tasep_ring_current():
    """Test that the parallel exclusion process on a ring keeps its particles and carries the exact current."""
    trajectory = Simulation(tasep(400, ring=True, p=0.5), random_state=0).run(until=1000).trajectory_
    occupied = trajectory.lattice('lane.occupied')
    current = ((occupied[:-1] == 1) & (occupied[1:] == 0)).sum(axis=1).mean() / 400
    assert set(occupied.sum(axis=1)) == {200}
    assert current == pytest.approx((1 - np.sqrt(0.5)) / 2, rel=0.03)


def test_tasep_open_line_stays_an_exclusion_process():
    """Test that the open exclusion process keeps one particle at most per site."""
    trajectory = Simulation(tasep(100), random_state=0).run(until=200).trajectory_
    assert set(np.unique(trajectory.lattice('lane.occupied'))) <= {0.0, 1.0}


def test_diffusion_limited_aggregate_is_connected():
    """Test that the aggregate grows and every stuck walker touches another stuck walker."""
    trajectory = Simulation(diffusion_limited_aggregation(200, size=31), random_state=0).run(until=100).trajectory_
    agents = trajectory.agents('walkers', 100)
    cells = {tuple(location) for location, stuck in zip(agents.locations, agents.values['stuck'], strict=True) if stuck}
    assert len(cells) > 20
    steps = ((0, 1), (1, 0), (0, -1), (-1, 0))
    assert all(any(((x + dx) % 31, (y + dy) % 31) in cells for dx, dy in steps) for x, y in cells)


def test_discrete_master_equation_relaxes_to_boltzmann():
    """Test that the master equation conserves probability and reaches `exp(-beta U) / Z`."""
    trajectory = Simulation(discrete_master_equation(), random_state=0).run(until=3000).trajectory_
    probability = trajectory.lattice('states.P')
    weights = np.exp(-trajectory.lattice('states.U')[-1])
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert np.allclose(probability[-1], weights / weights.sum(), atol=1e-6)


def test_two_phase_zone_exchange_is_binomial():
    """Test that the number of particles in a zone settles on the binomial mean and variance."""
    trajectory = Simulation(two_phase_zone_exchange(300), dt=0.5, random_state=0).run(until=12).trajectory_
    numbers = trajectory['zones.N'][trajectory.times >= 8]
    assert numbers.mean() == pytest.approx(10.0, rel=0.05)
    assert numbers.var() == pytest.approx(9.0, rel=0.15)


@pytest.mark.parametrize(('temperature', 'order'), [(0.5, np.sqrt(0.5)), (2.0, 0.0)])
def test_landau_relaxation_reaches_the_mean_field_order(temperature, order):
    """Test that the order parameter relaxes to `sqrt(a (Tc - T) / b)` below `Tc` and to 0 above it."""
    trajectory = Simulation(landau_relaxation(T=temperature), dt=0.1).run(until=50).trajectory_
    assert trajectory['magnet.phi'][-1] == pytest.approx(order, abs=1e-6)


def test_manna_sandpile_settles_below_two_grains():
    """Test that the Manna pile records avalanches, stays stable between grains, and fills to a critical density."""
    trajectory = Simulation(manna_sandpile((16, 16)), dt=100, random_state=0).run(until=3000).trajectory_
    heights = trajectory.lattice('pile.height')
    assert heights.max() < 2
    assert trajectory.events
    assert 0.55 < heights[-1].mean() < 0.75


def test_wave_breaking_steepens_after_the_breaking_time():
    """Test that the wave number stays smooth before the breaking time, steepens after it, and is conserved."""
    trajectory = Simulation(wave_breaking(), dt=0.5, solver=SSPRK()).run(until=12).trajectory_
    numbers = trajectory['train.k']
    jumps = np.abs(np.diff(numbers, axis=1)).max(axis=1)
    centers = (np.arange(200) + 0.5) * 2 * np.pi / 200
    initial = 1 + 0.5 * np.sin(centers)
    breaking = 1 / np.max(0.5 * np.cos(centers) / (4 * initial**1.5))
    assert jumps[trajectory.times < 0.7 * breaking].max() < 0.06
    assert jumps[-1] > 0.2
    assert numbers.sum(axis=1) == pytest.approx(np.full(len(numbers), numbers[0].sum()))


def test_discrete_velocity_boltzmann_conserves_mass_and_relaxes():
    """Test that the Carleman gas conserves its mass and relaxes to equal densities of both velocities."""
    trajectory = Simulation(discrete_velocity_boltzmann(), dt=0.1, solver=SSPRK()).run(until=5).trajectory_
    right, left = trajectory['gas.f_plus'], trajectory['gas.f_minus']
    assert (right + left).sum(axis=1) == pytest.approx(np.full(len(right), (right + left)[0].sum()))
    assert np.abs(right - left).max(axis=1)[-1] < 0.05 * np.abs(right - left).max(axis=1)[0]
