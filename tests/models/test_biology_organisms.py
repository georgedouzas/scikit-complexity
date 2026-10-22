"""Test the neuroscience, systems biology, physiology, collective behaviour, and oncology models."""

import math

import networkx as nx
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from skcomplex.core.analysis import autocorrelation
from skcomplex.core.base import ValidationError
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4, ScipyODE
from skcomplex.models.biology.collective_behaviour import (
    active_walkers,
    binary_ant_decision,
    bird_swarm,
    boids,
    rats,
)
from skcomplex.models.biology.neuroscience import (
    adaptrode,
    delayed_lif_network,
    fitzhugh_nagumo_ode,
    hodgkin_huxley,
    kinouchi_copelli,
    legion,
    polyhomeostatic_neuron,
    theta_neurons,
    wilson_cowan,
    winners_take_all,
)
from skcomplex.models.biology.oncology import tumour_angiogenesis
from skcomplex.models.biology.physiology import aliev_panfilov, cardiovascular
from skcomplex.models.biology.systems_biology import (
    circadian_clock,
    negative_autoregulation,
    random_boolean_network,
    repressilator,
    toggle_switch,
    yeast_cell_cycle,
)

HODGKIN_HUXLEY_THRESHOLD = 6.2
TOLERANCE = 0.1


def _autocorrelation_period(series, dt):
    """Return the period of a noisy oscillation, the lag of the first autocorrelation peak after its first zero."""
    correlation = autocorrelation(np.asarray(series) - np.mean(series))
    first_negative = int(np.argmax(correlation < 0))
    window = correlation[first_negative : first_negative + len(correlation) // 2]
    return (first_negative + int(np.argmax(window))) * dt


def _crossing_periods(times, series, level):
    """Return the intervals between upward crossings of a level."""
    above = np.asarray(series) >= level
    crossings = np.flatnonzero(~above[:-1] & above[1:])
    return np.diff(np.asarray(times)[crossings])


def _spike_times(trajectory):
    """Return the times of the spike events of a trajectory."""
    return np.array([event.time for event in trajectory.events if event.label == 'spike'])


def test_repressilator_oscillates_as_equations_and_jumps():
    """Test that the repressilator oscillates as ODEs and as a jump process, with periods within 10%."""
    periods = []
    for stochastic in (False, True):
        system = repressilator(stochastic=stochastic)
        trajectory = Simulation(system, dt=0.5, random_state=1).run(until=250).trajectory_
        protein = trajectory['gene_0.p'][trajectory.times > 50]
        assert protein.max() > 3 * max(protein.min(), 1.0)
        periods.append(_autocorrelation_period(protein, 0.5))
    deterministic, stochastic_period = periods
    assert abs(stochastic_period - deterministic) <= TOLERANCE * deterministic


def test_hodgkin_huxley_spikes():
    """Test that Hodgkin-Huxley fires periodically above a threshold current within 10% of the published 6.2."""
    trajectory = Simulation(hodgkin_huxley(I=10.0), dt=0.5).run(until=100).trajectory_
    intervals = np.diff(_spike_times(trajectory))[1:]
    assert len(intervals) >= 4
    assert np.ptp(intervals) < 0.01 * intervals.mean()
    below = Simulation(hodgkin_huxley(I=(1 - TOLERANCE) * HODGKIN_HUXLEY_THRESHOLD), dt=0.5).run(until=150)
    above = Simulation(hodgkin_huxley(I=(1 + TOLERANCE) * HODGKIN_HUXLEY_THRESHOLD), dt=0.5).run(until=150)
    assert len(_spike_times(below.trajectory_)) <= 1
    assert len(_spike_times(above.trajectory_)) >= 6


def test_toggle_switch_remembers_its_initial_state():
    """Test that the toggle switch settles to opposite states from opposite initial conditions."""
    finals = []
    for proteins in ((60.0, 0.0), (0.0, 60.0)):
        state = Simulation(toggle_switch(proteins), dt=1.0).run(until=100).trajectory_
        finals.append((state['gene_0.p'][-1], state['gene_1.p'][-1]))
    (first_high, first_low), (second_low, second_high) = finals
    assert first_high > 10 * first_low
    assert second_high > 10 * second_low


def test_negative_autoregulation_is_bursty():
    """Test that bursts make the protein noise super-Poissonian and that feedback lowers the mean."""
    statistics = []
    for K in (10.0, 1e9):
        trajectory = Simulation(negative_autoregulation(K=K), dt=1.0, random_state=0).run(until=400).trajectory_
        protein = trajectory['gene.p'][trajectory.times > 100]
        statistics.append((protein.mean(), protein.var() / protein.mean()))
    (regulated_mean, regulated_fano), (free_mean, free_fano) = statistics
    assert regulated_mean < 0.7 * free_mean
    assert free_fano > 3
    assert regulated_fano > 1


def test_circadian_clock_period_is_about_a_day():
    """Test that Goldbeter's PER model oscillates with a period of about 23.7 hours."""
    trajectory = Simulation(circadian_clock(), dt=0.5).run(until=240).trajectory_
    keep = trajectory.times > 48
    mrna = trajectory['clock.M'][keep]
    periods = _crossing_periods(trajectory.times[keep], mrna, mrna.mean())
    assert abs(periods.mean() - 23.7) < 1.0


def test_circadian_clock_period_lengthens_with_degradation():
    """Test that a larger maximal degradation rate of PER lengthens the period of Goldbeter's clock."""
    periods = []
    for v_d in (0.8, 1.6):
        trajectory = Simulation(circadian_clock(v_d=v_d), dt=0.1).run(until=480).trajectory_
        keep = trajectory.times > 240
        mrna = trajectory['clock.M'][keep]
        periods.append(_crossing_periods(trajectory.times[keep], mrna, mrna.mean()).mean())
    assert periods[0] == pytest.approx(22.6, abs=0.5)
    assert periods[1] == pytest.approx(29.3, abs=0.5)


def test_random_boolean_network_reaches_a_cycle():
    """Test that a random Boolean network returns to a state it visited, so it settles on an attractor."""
    states = Simulation(random_boolean_network(), dt=1).run(until=60).trajectory_.states
    seen = {tuple(row) for row in states[:30]}
    assert tuple(states[-1]) in seen or len({tuple(row) for row in states}) < len(states)


YEAST_RECORDED = (
    '10001000100',
    '01101000100',
    '01111000100',
    '01110000000',
    '01110001000',
    '01110001011',
    '00010011011',
    '00000110011',
    '00000110111',
    '00000110101',
    '00001110100',
    '00001100100',
    *('00001000100',) * 4,
)


def test_random_boolean_network_takes_eight_inputs():
    """Test that every node reads eight distinct nodes through a table of 256 entries, and the network runs."""
    system = random_boolean_network(n_nodes=12, k=8)
    assert all(len(set(function.inputs)) == 8 and len(function.table) == 256 for function in system.interactions)
    assert len(Simulation(system, dt=1).run(until=50).trajectory_.times) == 51


def test_random_boolean_network_rejects_more_inputs_than_nodes():
    """Test that a node cannot read more inputs than the network has nodes."""
    with pytest.raises(ValidationError, match=r'k = 5 .*n_nodes = 4'):
        random_boolean_network(n_nodes=4, k=5)


def test_yeast_cell_cycle_matches_recorded_trajectory():
    """Test that the yeast network on one variadic role per node runs as the five-role version recorded before."""
    states = Simulation(yeast_cell_cycle(), dt=1).run(until=15).trajectory_.states
    assert [''.join(str(int(value)) for value in row) for row in states] == list(YEAST_RECORDED)


def test_yeast_cell_cycle_returns_to_g1():
    """Test that the yeast network passes through the M phase and ends in the stationary G1 state."""
    simulation = Simulation(yeast_cell_cycle(), dt=1).run(until=15)
    names = simulation.problem_.states.names
    active = [[name for name, value in zip(names, row, strict=True) if value] for row in simulation.trajectory_.states]
    assert any('Clb1_2.x' in names for names in active)
    assert active[-1] == active[-2] == ['Cdh1.x', 'Sic1.x']


def test_fitzhugh_nagumo_neuron_needs_current_to_oscillate():
    """Test that the FitzHugh-Nagumo neuron oscillates with current and rests without."""
    amplitudes = []
    for current in (0.0, 0.5):
        trajectory = Simulation(fitzhugh_nagumo_ode(I=current), dt=0.5).run(until=200).trajectory_
        amplitudes.append(np.ptp(trajectory['neuron.v'][trajectory.times > 100]))
    assert amplitudes[0] < 0.01
    assert amplitudes[1] > 3


def test_wilson_cowan_limit_cycle():
    """Test that the Wilson-Cowan populations oscillate with input and decay to rest without."""
    amplitudes = []
    for drive in (0.0, 1.25):
        trajectory = Simulation(wilson_cowan(P=drive), dt=0.2).run(until=100).trajectory_
        amplitudes.append(np.ptp(trajectory['cortex.E'][trajectory.times > 50]))
    assert amplitudes[0] < 1e-6
    assert amplitudes[1] > 0.1


def test_theta_neuron_period():
    """Test that an uncoupled theta neuron with excitability `eta` fires with period `π / sqrt(eta)`."""
    system = theta_neurons(1, eta=1.0, delta=0.0, kappa=0.0)
    trajectory = Simulation(system, dt=0.05).run(until=20).trajectory_
    phase = np.unwrap(trajectory['neurons.theta'][:, 0])
    turns = (phase[-1] - phase[0]) / (2 * math.pi)
    assert abs(20 / turns - math.pi) < 0.05


def _ott_antonsen(z0, times, *, eta=-0.2, delta=0.1, kappa=1.5, n=2):
    """Return the order parameter of the Ott-Antonsen reduction of Luke, Barreto, and So for theta neurons."""
    normalization = 2**n * math.factorial(n) ** 2 / math.factorial(2 * n)
    grid = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    coefficients = np.fft.fft((1 - np.cos(grid)) ** n).real / grid.size

    def pulses(z):
        return normalization * (coefficients[0] + sum(coefficients[k] * 2 * (z**k).real for k in range(1, n + 1)))

    def rates(t, y):
        z = y[0] + 1j * y[1]
        rate = -1j * (z - 1) ** 2 / 2 + (z + 1) ** 2 / 2 * (-delta + 1j * eta + 1j * kappa * pulses(z))
        return [rate.real, rate.imag]

    solution = solve_ivp(rates, (times[0], times[-1]), [z0.real, z0.imag], t_eval=times, rtol=1e-9, atol=1e-11)
    return solution.y[0] + 1j * solution.y[1]


def test_theta_neurons_follow_the_ott_antonsen_reduction():
    """Test that 2000 theta neurons reading their mean pulse follow the Ott-Antonsen reduction after the transient."""
    system = theta_neurons(2000)
    trajectory = Simulation(system, dt=1.0, solver=ScipyODE()).run(until=40).trajectory_
    assert trajectory.problem.states.names == ['neurons.theta']
    order = np.exp(1j * trajectory['neurons.theta']).mean(axis=1)
    reduced = _ott_antonsen(order[0], trajectory.times)
    assert np.abs(order - reduced)[trajectory.times >= 20].max() < 0.05


def test_winners_take_all_with_one_inhibition():
    """Test that the default network has one inhibition reading the total, and only the unit with input 1.1 wins."""
    for inputs in ((1.0, 1.1, 0.9), (1.0, 1.1, 0.9, 1.05, 0.95)):
        assert len(winners_take_all(inputs).interactions) == 1
    state = Simulation(winners_take_all(beta=2.0), dt=1.0).run(until=200).state_
    assert np.argmax(state) == 1
    assert abs(state[1] - 1.1) < 1e-3
    assert np.sort(state)[-2] < 1e-3


def test_winner_takes_all():
    """Test that strong inhibition leaves only the unit with the largest input, and weak inhibition several."""
    inputs = (1.0, 1.1, 0.9)
    strong = Simulation(winners_take_all(inputs, beta=2.0), dt=1.0).run(until=80).state_
    weak = Simulation(winners_take_all(inputs, beta=0.5), dt=1.0).run(until=80).state_
    assert np.argmax(strong) == 1
    assert strong[1] == pytest.approx(1.1, rel=1e-3)
    assert np.sort(strong)[-2] < 1e-3
    assert np.all(weak > 0.1)


def test_polyhomeostatic_neuron_approaches_target_rate():
    """Test that adaptation brings the mean firing rate of the polyhomeostatic neuron to its target."""
    trajectory = Simulation(polyhomeostatic_neuron(mu=0.2), dt=0.5).run(until=600).trajectory_
    x, a, b = trajectory['neuron.x'], trajectory['neuron.a'], trajectory['neuron.b']
    rate = 1 / (1 + np.exp(a * (b - x)))
    assert abs(rate[trajectory.times > 400].mean() - 0.2) < 0.05
    fixed = Simulation(polyhomeostatic_neuron(eps_a=0.0, eps_b=0.0), dt=0.5).run(until=100).trajectory_
    fixed_rate = 1 / (1 + np.exp(-fixed['neuron.x'][fixed.times > 50]))
    assert fixed_rate.mean() == pytest.approx(0.5, abs=0.02)


def test_delayed_lif_network_rates():
    """Test that uncoupled neurons fire at the leaky integrate-and-fire period and that inhibition slows them."""
    period = 20.0 * math.log((25.0 - 10.0) / (25.0 - 20.0))
    counts = []
    for J in (0.0, -2.0):
        simulation = Simulation(delayed_lif_network(5, J=J, probability=0.5), dt=1.0).run(until=60)
        events = simulation.trajectory_.events
        counts.append(len(events))
        if J == 0.0:
            index = list(simulation.problem_.states.names).index('neuron_0.v')
            times = [event.time for event in events if event.before[index] != event.after[index]]
            assert np.diff(times) == pytest.approx(period, rel=1e-3)
    assert counts[1] < counts[0]


def test_legion_segments_objects():
    """Test that LEGION synchronizes each object, alternates between objects, and keeps the background silent."""
    trajectory = Simulation(legion(epsilon=0.1), dt=0.5).run(until=120).trajectory_
    active = trajectory['oscillators.x'].reshape(len(trajectory.times), -1) > 0
    first, second = active[:, [0, 1, 4]], active[:, [11, 15]]
    assert first.mean() > 0.1
    assert second.mean() > 0.1
    assert (first[:, 0] == first[:, 1]).mean() > 0.97
    assert (second[:, 0] == second[:, 1]).mean() > 0.97
    after = trajectory.times > 20
    assert not (first.any(axis=1) & second.any(axis=1))[after].any()
    assert first[after].any()
    assert second[after].any()
    assert active[:, [2, 3, 5, 6, 7]].mean() < 0.02


def test_legion_defaults_fire_both_objects_in_a_short_run():
    """Test that both objects of the default LEGION network start an active phase within 100 time units."""
    trajectory = Simulation(legion(), dt=0.5).run(until=100).trajectory_
    active = trajectory['oscillators.x'].reshape(len(trajectory.times), -1) > 0
    for sites in ([0, 1, 4], [11, 15]):
        on = active[:, sites].any(axis=1)
        assert (on[1:] & ~on[:-1]).any()


def _dynamic_range(rates, response):
    """Return the dynamic range in dB of a response curve, `10 log10(r_0.9 / r_0.1)`."""
    response = np.maximum.accumulate(response)
    low, high = response[0], response[-1]
    stimulus = [np.exp(np.interp(low + share * (high - low), response, np.log(rates))) for share in (0.1, 0.9)]
    return 10 * np.log10(stimulus[1] / stimulus[0])


def test_kinouchi_copelli_activity_dies_out_below_and_persists_above_the_critical_point():
    """Test that without stimulus the activity dies out below a branching ratio of 1 and sustains itself above."""
    graph = nx.gnp_random_graph(2000, 10 / 1999, seed=0)
    for sigma, persists in ((0.8, False), (1.5, True)):
        trajectory = Simulation(kinouchi_copelli(graph, sigma=sigma), dt=1, random_state=0).run(300).trajectory_
        assert (trajectory['neurons.phase'][-50:] == 1).any() == persists


def test_kinouchi_copelli_dynamic_range_peaks_at_the_critical_point():
    """Test that the dynamic range of the response to the stimulus peaks at a branching ratio of 1."""
    graph = nx.gnp_random_graph(2000, 10 / 1999, seed=0)
    rates = np.logspace(-5, 1, 16)
    sigmas = (0.5, 0.75, 1.0, 1.25, 1.5)
    ranges = []
    for sigma in sigmas:
        response = []
        for rate in rates:
            system = kinouchi_copelli(graph, sigma=sigma, stimulus=rate)
            trajectory = Simulation(system, dt=1, random_state=0).run(1000).trajectory_
            response.append((trajectory['neurons.phase'][200:] == 1).mean())
        ranges.append(_dynamic_range(rates, np.array(response)))
    assert sigmas[int(np.argmax(ranges))] == 1.0
    assert min(ranges[2] - ranges[1], ranges[2] - ranges[3]) > 1.0


def test_adaptrode_keeps_long_term_memory():
    """Test that the Adaptrode weights order by time scale during input and that the slow weight outlasts it."""
    trajectory = Simulation(adaptrode(), dt=1).run(until=400).trajectory_
    w0, w1, w2 = (trajectory[f'synapse.w{index}'] for index in range(3))
    during = (trajectory.times > 100) & (trajectory.times < 150)
    assert np.all(w0[during] > w1[during])
    assert np.all(w1[during] > w2[during])
    after = trajectory.times == 200
    assert w0[after][0] - w1[after][0] < 0.05
    assert w2[-1] > 0.5 * w2[trajectory.times == 150][0]


def test_aliev_panfilov_sustains_a_spiral():
    """Test that the broken wave in the Aliev-Panfilov tissue keeps the tissue active, as a re-entrant spiral."""
    simulation = Simulation(aliev_panfilov((50, 50)), dt=50.0, solver=RK4(dt=0.05)).run(until=400)
    excited = (simulation.trajectory_['tissue.u'] > 0.5).mean(axis=1)
    assert np.all(excited[1:] > 0.05)


def test_cardiovascular_conserves_blood_and_pulses():
    """Test that the circulation conserves the volume of blood and gives an arterial pressure near 120/80 mmHg."""
    trajectory = Simulation(cardiovascular(), dt=0.02).run(until=6.4).trajectory_
    total = trajectory['ventricle.V'] + trajectory['arteries.V'] + trajectory['veins.V']
    assert np.ptp(total) < 1e-6 * total[0]
    pressure = (trajectory['arteries.V'][trajectory.times > 4.8] - 250.0) * 1.3**-1
    assert 65 < pressure.min() < 95
    assert 105 < pressure.max() < 135


def test_boids_align():
    """Test that boids starting with random headings become a polarized flock."""
    simulation = Simulation(boids(), dt=1, random_state=0).run(until=100)

    def polarization(frame):
        snapshot = simulation.trajectory_.agents('flock', frame)
        velocity = np.stack([snapshot.values['vx'], snapshot.values['vy']], axis=1)
        return np.hypot(*(velocity / np.hypot(velocity[:, :1], velocity[:, 1:])).mean(axis=0))

    assert polarization(0) < 0.3
    assert polarization(100) > 0.7


def test_bird_swarm_flies_together():
    """Test that the swarm reaches the speed `sqrt(alpha / beta)` and stays together."""
    trajectory = Simulation(bird_swarm(), dt=0.5).run(until=50).trajectory_
    positions, velocities = trajectory['birds.x'][-1], trajectory['birds.v'][-1]
    speeds = np.hypot(velocities[:, 0], velocities[:, 1])
    assert np.median(speeds) == pytest.approx(math.sqrt(2.0), rel=0.1)
    assert np.hypot(*(positions - positions.mean(axis=0)).T).max() < 10.0


def test_binary_ant_decision_breaks_symmetry():
    """Test that ants pick one branch in most stochastic runs, while the deterministic mode stays symmetric."""
    shares = []
    for seed in range(5):
        state = Simulation(binary_ant_decision(), dt=10.0, random_state=seed).run(until=1000).state_
        shares.append(state.max() / state.sum())
    assert sum(share > 0.75 for share in shares) >= 4
    state = Simulation(binary_ant_decision(stochastic=False), dt=10.0).run(until=1000).state_
    assert state[0] == pytest.approx(state[1])


def test_rats_phase_transition():
    """Test that nervousness dies out below the transition and settles at `1 - calming / contagion` above it."""
    below = Simulation(rats(spontaneous=0.0, contagion=0.5, stochastic=False), dt=1.0).run(until=50).state_
    above = Simulation(rats(spontaneous=0.0, contagion=2.0, stochastic=False), dt=1.0).run(until=50).state_
    assert below[1] < 0.1
    assert above[1] / above.sum() == pytest.approx(0.5, rel=1e-3)


def test_active_walkers_bundle_trails():
    """Test that attraction to trails makes walkers wear fewer cells of the green than walkers who ignore trails."""
    worn = []
    for attraction in (0.0, 1.0):
        simulation = Simulation(active_walkers(attraction=attraction), dt=1, random_state=0)
        trajectory = simulation.run(until=300).trajectory_
        worn.append(int((trajectory['land/ground.G'][-1] > 0.5).sum()))
    assert worn[1] < 0.8 * worn[0]


def test_tumour_recruits_vessels_and_grows():
    """Test that vessel tips move toward the hypoxic tumour, which then grows."""
    trajectory = Simulation(tumour_angiogenesis(), dt=1, random_state=0).run(until=120).trajectory_
    assert trajectory.agents('cells/tumour', 50).values['hypoxic'].all()
    assert trajectory.agents('cells/sprouts', 120).locations[:, 0].mean() > 15
    assert len(trajectory.agents('cells/tumour', 120)) > 2 * len(trajectory.agents('cells/tumour', 0))
