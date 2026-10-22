"""Test the Bertin agents, evolution, and population models: choices, standardization, fixation, and neutral ecology."""

import itertools

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import gammaln

from skcomplex.core.runtime import Simulation
from skcomplex.models import describe
from skcomplex.models.biology.ecology import neutral_biodiversity, neutral_clustering
from skcomplex.models.biology.epidemiology import heterogeneous_mean_field_sir
from skcomplex.models.biology.evolution import origin_fixation_walk
from skcomplex.models.economics.markets import perishable_goods_duopoly
from skcomplex.models.social.opinion_dynamics import standardized_agents


def _degree_moments(exponent, k_min, k_max):
    """Return the degrees, their probabilities, and the first two moments of a power-law degree distribution."""
    degrees = np.arange(k_min, k_max + 1, dtype=float)
    probabilities = degrees**-exponent / np.sum(degrees**-exponent)
    return degrees, probabilities, np.sum(degrees * probabilities), np.sum(degrees**2 * probabilities)


def _expected_sample_variance(sites, alpha, beta, n, times):
    """Return the expected variance over the sites of a ring, from the exact moment equations of neutral clustering."""

    def derivative(t, covariance):
        rates = 2 * beta * (np.roll(covariance, 1) + np.roll(covariance, -1) - 2 * covariance)
        rates[0] += 2 * alpha * n
        return rates

    initial = np.zeros(sites)
    initial[0] = -n
    solution = solve_ivp(derivative, (0, times[-1]), initial, t_eval=times, rtol=1e-8).y
    covariance = solution + n * (np.arange(sites) == 0)[:, None]
    return covariance[0] - covariance.mean(axis=0)


@pytest.mark.parametrize(('factor', 'below'), [(0.5, True), (2.0, False)])
def test_heterogeneous_mean_field_sir_epidemic_threshold(factor, below):
    """Test that the final epidemic size is below 1% below the threshold <k>/<k^2> and above 5% above it."""
    _, _, first, second = _degree_moments(2.5, 3, 100)
    beta = factor * first / second
    trajectory = (
        Simulation(
            built := heterogeneous_mean_field_sir(beta=beta, mu=1.0),
            dt=1.0,
            observables=describe('heterogeneous_mean_field_sir').observables_of(built),
        )
        .run(until=200)
        .trajectory_
    )
    final_size = trajectory['recovered'][-1]
    assert trajectory['infected'][-1] < 1e-6
    assert (final_size < 0.01) if below else (final_size > 0.05)


def test_heterogeneous_mean_field_sir_early_growth_rate():
    """Test that theta grows at the rate (beta (<k^2> - <k>) - mu <k>) / <k> of the linearized equation (6.30)."""
    degrees, probabilities, first, second = _degree_moments(2.5, 3, 100)
    beta, mu = 0.13, 1.0
    system = heterogeneous_mean_field_sir(beta=beta, mu=mu, infected=1e-9)
    trajectory = (
        Simulation(system, dt=0.5, observables=describe('heterogeneous_mean_field_sir').observables_of(system))
        .run(until=12)
        .trajectory_
    )
    theta = trajectory['network.i'] @ ((degrees - 1) * probabilities) / first
    selected = trajectory.times >= 4
    rate = np.polyfit(trajectory.times[selected], np.log(theta[selected]), 1)[0]
    assert rate == pytest.approx((beta * (second - first) - mu * first) / first, rel=0.03)


def test_origin_fixation_walk_samples_fitness_to_the_power_n_minus_one():
    """Test that the time spent on each genome of four loci matches the stationary law Q ~ f^(N - 1)."""
    loci, size = 4, 10.0
    system = origin_fixation_walk(loci, N=size, mu=1.0, field=0.1, coupling=0.3, random_state=1)
    trajectory = (
        Simulation(system, dt=1.0, random_state=0, observables=describe('origin_fixation_walk').observables_of(system))
        .run(until=5000)
        .trajectory_
    )
    alleles = trajectory['genome.s'][100:]
    visited = np.bincount(((alleles > 0) * 2 ** np.arange(loci)).sum(axis=1).astype(int), minlength=2**loci)
    fields = system.elements[0].values['h']
    couplings = system.interactions[0].couplings
    genomes = np.array([[2 * bit - 1 for bit in reversed(bits)] for bits in itertools.product((0, 1), repeat=loci)])
    fitness = 1.0 + genomes @ fields + np.einsum('gi,ij,gj->g', genomes, couplings, genomes) / 2
    assert visited / visited.sum() == pytest.approx(fitness ** (size - 1) / np.sum(fitness ** (size - 1)), abs=0.02)


def test_origin_fixation_walk_climbs_the_landscape():
    """Test that a large population ends near the single peak of a landscape of random fields."""
    system = origin_fixation_walk(16, N=1000.0, mu=0.01, random_state=0)
    trajectory = (
        Simulation(system, dt=10.0, random_state=0, observables=describe('origin_fixation_walk').observables_of(system))
        .run(until=1000)
        .trajectory_
    )
    fields = system.elements[0].values['h']
    assert trajectory['fitness'][-1] == pytest.approx(1.0 + np.abs(fields).sum(), abs=0.02)
    assert trajectory['fitness'][-1] > trajectory['fitness'][0]


def test_neutral_biodiversity_species_number_matches_ewens_sum():
    """Test that the mean number of species matches S_N = sum_n theta / (theta + n) within 5%."""
    size, nu = 100, 0.02
    theta = (size - 1) * nu / (1 - nu)
    trajectory = (
        Simulation(
            built := neutral_biodiversity(size, nu=nu, random_state=0),
            random_state=0,
            observables=describe('neutral_biodiversity').observables_of(built),
        )
        .run(until=3000)
        .trajectory_
    )
    expected = np.sum(theta / (theta + np.arange(size)))
    assert trajectory['richness'][500:].mean() == pytest.approx(expected, rel=0.05)


def test_neutral_clustering_variance_grows_linearly_without_hops():
    """Test that without hops the variance over 400 patches, averaged over six runs, grows as 2 alpha n t."""
    system = neutral_clustering(400, n=10.0, alpha=1.0, beta=0.0)
    runs = [
        Simulation(
            system,
            dt=2.0,
            random_state=seed,
            backend='numba',
            observables=describe('neutral_clustering').observables_of(system),
        )
        .run(until=10)
        .trajectory_
        for seed in range(6)
    ]
    times = runs[0].times
    assert np.mean([run['mean'][-1] for run in runs]) == pytest.approx(10.0, rel=0.1)
    assert np.mean([run['variance'] for run in runs], axis=0)[1:] == pytest.approx(2 * 10.0 * times[1:], rel=0.2)


def test_neutral_clustering_hops_slow_but_keep_the_clustering():
    """Test that hops on a ring give the variance of the exact moment equations, which grows below 2 alpha n t."""
    sites, until = 64, 20.0
    system = neutral_clustering(sites, n=10.0, alpha=1.0, beta=1.0)
    variances = [
        Simulation(
            system,
            dt=5.0,
            random_state=seed,
            backend='numba',
            observables=describe('neutral_clustering').observables_of(system),
        )
        .run(until=until)
        .trajectory_['variance']
        for seed in range(32)
    ]
    times = np.arange(0.0, until + 1, 5.0)
    expected = _expected_sample_variance(sites, 1.0, 1.0, 10.0, times)
    assert np.mean(variances, axis=0)[1:] == pytest.approx(expected[1:], rel=0.25)
    assert expected[-1] < 0.5 * 2 * 10.0 * until
    assert expected[-1] > 1.5 * expected[1]


def _asymmetry_at_mean_field(temperature, control):
    """Return the stable asymmetry of the mean-field equation (4.48) of the perishable goods model."""

    def drift(m):
        return temperature * np.log((1 - m) / (1 + m)) + 1 / (1 + control / (1 + m)) - 1 / (1 + control / (1 - m))

    return brentq(drift, 1e-6, 1 - 1e-12)


@pytest.mark.parametrize('ratio', [0.5, 2.0])
def test_perishable_goods_duopoly_symmetry_breaking(ratio):
    """Test that one shop takes most customers below T_c = R / (1 + R)^2 and that both shops share them above it."""
    control = 1.0
    critical = control / (1 + control) ** 2
    system = perishable_goods_duopoly(200, stock=0.5, tau_a=1.0, T=ratio * critical, random_state=0)
    asymmetry = np.abs(
        Simulation(system, random_state=0, observables=describe('perishable_goods_duopoly').observables_of(system))
        .run(until=300)
        .trajectory_['m'][-50:],
    ).mean()
    if ratio < 1:
        assert asymmetry == pytest.approx(_asymmetry_at_mean_field(ratio * critical, control), abs=0.1)
    else:
        assert asymmetry < 0.05


def test_standardized_agents_match_the_exact_equilibrium():
    """Test that the time-averaged share q of ten agents matches the exact distribution P ~ exp(E / T)."""
    size, configurations, K = 10, 20, 4.0
    temperature = K / (2 * np.log(configurations / 2))
    marked = np.arange(size + 1)
    weights = gammaln(size + 1) - gammaln(marked + 1) - gammaln(size - marked + 1)
    weights += marked * np.log(2) + (size - marked) * np.log(configurations)
    weights += K * marked * (marked - 1) / (2 * size * temperature)
    probabilities = np.exp(weights - weights.max()) / np.exp(weights - weights.max()).sum()
    system = standardized_agents(size, K=K, J=0.0, T=temperature, H=configurations, standardized=0.0, random_state=0)
    q = (
        Simulation(system, random_state=0, observables=describe('standardized_agents').observables_of(system))
        .run(until=5000)
        .trajectory_['q']
    )
    assert q[200:].mean() == pytest.approx(np.sum(probabilities * marked / size), abs=0.04)


@pytest.mark.parametrize(('temperature', 'ordered'), [(0.5, True), (1.3, False)])
def test_standardized_agents_order_below_t_m(temperature, ordered):
    """Test that standardized agents magnetize below T_M = J, to the root of m = tanh(J m / T), and not above."""
    system = standardized_agents(100, K=10.0, J=1.0, T=temperature, H=50, random_state=0)
    trajectory = (
        Simulation(system, random_state=0, observables=describe('standardized_agents').observables_of(system))
        .run(until=1000)
        .trajectory_
    )
    magnetization = np.abs(trajectory['m'][-200:]).mean()
    assert trajectory['q'][-200:].mean() > 0.95
    if ordered:
        assert magnetization == pytest.approx(brentq(lambda m: m - np.tanh(m / temperature), 0.1, 1.0), abs=0.08)
    else:
        assert magnetization < 0.2
