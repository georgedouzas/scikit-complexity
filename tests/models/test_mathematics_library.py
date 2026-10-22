"""Test the mathematics models: dynamical systems, stochastic processes, and network processes."""

import math

import networkx as nx
import numpy as np
import pytest
from scipy import stats

from skcomplex.core.analysis import lyapunov_spectrum
from skcomplex.core.runtime import Simulation
from skcomplex.models import describe
from skcomplex.models.mathematics.dynamical_systems import (
    baker_map,
    cat_map,
    circle_map,
    cusp_form,
    finite_time_blow_up,
    hopf_form,
    intermittency_map,
    koopman_slow_manifold,
    newton_raphson_map,
    ou_driven_rossler,
    saddle_node_form,
    standard_map,
    takens_bogdanov_form,
    tent_map,
    towel_map,
)
from skcomplex.models.mathematics.fractals import julia, mandelbrot
from skcomplex.models.mathematics.networks import (
    network_diffusion,
    network_random_walk,
    network_structure_analysis,
    pagerank,
)
from skcomplex.models.mathematics.stochastic_processes import (
    ageing_random_walk,
    bernoulli_process,
    fragmentation,
    galton_watson,
    poisson_process,
    polya_urn,
    random_walk,
    sample_space_reducing,
    weakening_ornstein_uhlenbeck,
)

GOLDEN_MEAN = (math.sqrt(5) - 1) / 2
CRITICAL_KICK = 0.971635


def _run(system, until, dt=1, random_state=0, observables=()):
    """Return the trajectory of a system run with a frame duration and a random state."""
    simulation = Simulation(system, dt=dt, random_state=random_state, observables=observables)
    return simulation.run(until=until).trajectory_


def _momentum_range(K, theta, iterations):
    """Return the spread of the momentum of a standard map orbit started near the golden-mean torus."""
    momentum = _run(standard_map(theta=theta, p=2 * math.pi * GOLDEN_MEAN, K=K), iterations)['map.p']
    return momentum.max() - momentum.min()


def test_standard_map_diffuses_above_the_critical_kick():
    """Test that a standard map orbit diffuses in momentum above K = 0.971635 and stays bounded below it."""
    assert _momentum_range(CRITICAL_KICK + 0.23, 0.0, 20000) > 4 * math.pi
    assert all(_momentum_range(CRITICAL_KICK - 0.02, theta, 10000) < 2 * math.pi for theta in (0.0, math.pi / 2))


def test_circle_map_rotates_rationally_without_nonlinearity():
    """Test that the circle map without nonlinearity is a rotation of period 4 for a quarter turn."""
    theta = _run(circle_map(0.1, Omega=0.25, K=0.0), 8)['map.theta']
    np.testing.assert_allclose(theta[4], theta[0], atol=1e-12)
    np.testing.assert_allclose(theta[1], 0.35, atol=1e-12)


def test_tent_map_decays_below_unit_slope_and_stays_in_its_interval_above():
    """Test that the tent map tends to 0 for a slope below 1 and stays in `[0, μ / 2]` for a chaotic slope."""
    assert _run(tent_map(0.3, mu=0.8), 100)['map.x'][-1] < 1e-8
    x = _run(tent_map(0.3, mu=1.9), 500)['map.x']
    assert x.min() >= 0
    assert x.max() <= 0.95 + 1e-12


def test_baker_map_leaves_a_gap_when_it_contracts():
    """Test that the dissipative baker map leaves the band between its two images empty."""
    y = _run(baker_map(0.2, 0.5, alpha=0.4, lambda_a=0.3, lambda_b=0.3), 300)['map.y'][20:]
    assert not np.any((y > 0.3 + 1e-9) & (y < 0.7 - 1e-9))
    y = _run(baker_map(0.2, 0.5), 300)['map.y'][20:]
    assert np.any((y > 0.4) & (y < 0.6))


def test_towel_map_is_hyperchaotic():
    """Test that Rössler's towel map has two positive Lyapunov exponents."""
    exponents = lyapunov_spectrum(towel_map(), until=5000)
    assert exponents[1] > 0.1
    assert exponents[2] < 0


def test_intermittency_map_has_a_long_laminar_phase():
    """Test that the Pomeau-Manneville map lingers near 0 for about `1 / x0` steps."""
    x = _run(intermittency_map(0.01, z=2.0), 300)['map.x']
    assert 60 < np.argmax(x > 0.1) < 120


def test_newton_raphson_map_finds_the_cube_roots_of_unity():
    """Test that Newton's method converges to the cube root of unity whose basin holds the start."""
    first = _run(newton_raphson_map(1.5, 0.1), 50)
    np.testing.assert_allclose([first['map.x'][-1], first['map.y'][-1]], [1.0, 0.0], atol=1e-10)
    second = _run(newton_raphson_map(-1.0, 1.0), 50)
    np.testing.assert_allclose([second['map.x'][-1], second['map.y'][-1]], [-0.5, math.sqrt(3) / 2], atol=1e-10)


def test_cat_map_has_a_periodic_orbit_of_period_three():
    """Test that the cat map sends `(1/2, 1/2)` around an orbit of period 3."""
    trajectory = _run(cat_map(0.5, 0.5), 3)
    np.testing.assert_allclose([trajectory['map.x'][3], trajectory['map.y'][3]], [0.5, 0.5])
    assert trajectory['map.y'][1] == 0.0


def test_normal_forms_settle_on_their_attractors():
    """Test the stable node of the saddle-node, the Hopf cycle, the Takens-Bogdanov focus, and the cusp bistability."""
    assert _run(saddle_node_form(0.0, mu=1.0), 20, dt=0.1)['form.x'][-1] == pytest.approx(1.0, abs=1e-4)
    hopf = _run(hopf_form(mu=0.25), 60, dt=0.1)
    assert math.hypot(hopf['form.x'][-1], hopf['form.y'][-1]) == pytest.approx(0.5, abs=1e-3)
    focus = (1 - math.sqrt(0.6)) / 2
    assert _run(takens_bogdanov_form(), 200, dt=0.5)['form.x'][-1] == pytest.approx(focus, abs=1e-3)
    assert _run(cusp_form(0.5), 30, dt=0.5)['form.x'][-1] == pytest.approx(1.0, abs=1e-4)
    assert _run(cusp_form(-0.5), 30, dt=0.5)['form.x'][-1] == pytest.approx(-1.0, abs=1e-4)


def test_finite_time_blow_up_follows_the_hyperbola():
    """Test that `x' = x²` follows `x0 / (1 - x0 t)` towards its singularity at `t = 1 / x0`."""
    assert _run(finite_time_blow_up(1.0), 0.9, dt=0.05)['growth.x'][-1] == pytest.approx(10.0, rel=1e-4)


def test_random_walk_spreads_diffusively_and_drifts_with_bias():
    """Test that the variance of unbiased walkers grows as `rate t` and biased walkers drift as `rate (2 bias - 1)`."""
    positions = _run(random_walk(500), 50)['walkers.x'][-1]
    assert abs(positions.mean()) < 1.0
    assert positions.var() == pytest.approx(50, rel=0.15)
    assert _run(random_walk(500, bias=0.75), 50)['walkers.x'][-1].mean() == pytest.approx(25, rel=0.1)


def test_weakening_ornstein_uhlenbeck_has_student_tails():
    """Test that the stationary density is Student's t with `2 nu - 1` degrees of freedom, `nu = k ℓ² / 2D`."""
    positions = _run(weakening_ornstein_uhlenbeck(500, D=0.25), 20, dt=0.1)['particles.x'][-100:]
    expected = stats.t(df=3).ppf(0.95) / math.sqrt(3)
    assert np.quantile(np.abs(positions), 0.9) == pytest.approx(expected, rel=0.1)


def test_polya_urn_shares_tend_to_a_uniform_limit():
    """Test that the red shares of urns started with one ball of each color are uniform with mean 1/2."""
    trajectory = _run(polya_urn(300), 60)
    red, blue = trajectory['urns.red'][-1], trajectory['urns.blue'][-1]
    assert (red + blue).mean() == pytest.approx(62, rel=0.05)
    shares = red / (red + blue)
    assert shares.mean() == pytest.approx(0.5, abs=0.05)
    assert stats.kstest(shares, 'uniform').pvalue > 0.01


def test_galton_watson_grows_geometrically_on_average():
    """Test that the mean generation size is `Z0 m^n`, and that subcritical processes die out."""
    sizes = [_run(galton_watson(50, m=1.2), 5, random_state=seed)['population.count'][-1] for seed in range(30)]
    assert np.mean(sizes) == pytest.approx(50 * 1.2**5, rel=0.15)
    assert _run(galton_watson(10, m=0.5), 30)['population.count'][-1] == 0


def test_sample_space_reducing_visits_follow_zipf():
    """Test that a sample-space-reducing process visits state `i` with a frequency proportional to `1 / i`."""
    trajectory = _run(sample_space_reducing(200, n_states=10), 200)
    visits = np.concatenate([trajectory.agents('balls', frame).values['level'] for frame in range(1, 201)])
    counts = np.bincount(visits.astype(int), minlength=11)
    assert counts[1] / counts[2] == pytest.approx(2, rel=0.1)
    assert counts[1] / counts[5] == pytest.approx(5, rel=0.15)


def test_fragmentation_conserves_the_total_size():
    """Test that fragments keep breaking while their total size stays that of the initial object."""
    trajectory = _run(fragmentation(500), 20)
    lengths = trajectory.agents('fragments', 20).values['length']
    assert lengths.sum() == 500
    assert trajectory['fragments.count'][-1] > trajectory['fragments.count'][5] > 1


def test_ageing_random_walk_slows_its_up_steps():
    """Test that up steps grow linearly without ageing and as the square root of the steps for `nu = 1`."""
    ups = _run(ageing_random_walk(200, nu=0.0), 100).agents('walkers', 100).values['ups']
    assert ups.mean() == pytest.approx(50, rel=0.05)
    ups = _run(ageing_random_walk(200, nu=1.0), 100).agents('walkers', 100).values['ups']
    assert ups.mean() == pytest.approx(math.sqrt(100) - 1, rel=0.2)


def test_bernoulli_and_poisson_processes_count_at_their_rates():
    """Test that Bernoulli successes average `p n` and Poisson counts average and vary as `rate t`."""
    successes = _run(bernoulli_process(100, p=0.3), 100).agents('trials', 100).values['successes']
    assert successes.mean() == pytest.approx(30, rel=0.05)
    counts = _run(poisson_process(500, rate=2.0), 20)['counters.count'][-1]
    assert counts.mean() == pytest.approx(40, rel=0.05)
    assert counts.var() == pytest.approx(40, rel=0.2)


def test_network_random_walk_visits_nodes_in_proportion_to_degree():
    """Test that random walkers occupy the nodes of a connected graph in proportion to their degrees."""
    graph = nx.karate_club_graph()
    trajectory = _run(network_random_walk(graph, n_walkers=200), 100)
    nodes = np.concatenate([trajectory.agents('walkers', frame).locations for frame in range(20, 101)])
    visits = np.bincount(np.ravel(nodes).astype(int), minlength=graph.number_of_nodes())
    degrees = np.array([degree for _, degree in graph.degree()])
    assert np.corrcoef(visits, degrees)[0, 1] > 0.95


def test_network_diffusion_conserves_mass_and_equalizes():
    """Test that diffusion on a connected network conserves the total and tends to the uniform content."""
    contents = _run(network_diffusion(), 40, dt=1.0)['nodes.u']
    np.testing.assert_allclose(contents.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(contents[-1], 1 / 34, atol=1e-4)


def test_pagerank_matches_networkx():
    """Test that the PageRank iteration converges to the PageRank computed by NetworkX."""
    graph = nx.les_miserables_graph()
    ranks = _run(pagerank(graph, damping=0.85), 200)['pages.rank'][-1]
    expected = nx.pagerank(graph, alpha=0.85, weight=None, tol=1e-14, max_iter=1000)
    np.testing.assert_allclose(ranks, [expected[node] for node in graph.nodes], atol=1e-8)


def test_mandelbrot_escape_times():
    """Test that `c = 1` escapes at step 3 and that `c = -1`, `c = -2`, and `c = i` never escape in 100 steps."""
    escapes = _run(mandelbrot((5, 5), window=(-2, 2, -2, 2)), 100).lattice('plane.n')[-1]
    assert escapes[2, 3] == 3
    assert escapes[2, 1] == escapes[2, 0] == escapes[1, 2] == 100


def test_julia_set_of_squaring_map_is_unit_disk():
    """Test that the filled Julia set of `z -> z^2` keeps the points inside the unit circle and loses those outside."""
    trajectory = _run(julia(0.0, (201, 201), window=(-2, 2, -2, 2)), 100)
    modulus, escapes = np.abs(trajectory.lattice('plane.z')[0]), trajectory.lattice('plane.n')[-1]
    assert np.all(escapes[modulus < 0.99] == 100)
    assert np.all(escapes[modulus > 1.01] < 100)


def test_mandelbrot_is_symmetric_about_real_axis():
    """Test that the escape times on a window symmetric about the real axis equal their mirror image."""
    escapes = _run(mandelbrot((61, 80), window=(-2.2, 0.8, -1.2, 1.2)), 60).lattice('plane.n')[-1]
    assert np.array_equal(escapes, escapes[::-1])
    assert escapes.min() < escapes.max()


def test_network_structure_analysis_of_a_generated_network():
    """Test that a planted partition of 4 blocks of 100 nodes is recorded with its measures and its blocks found."""
    trajectory = _run(
        built := network_structure_analysis(null_samples=5),
        1,
        observables=describe('network_structure_analysis').observables_of(built),
    )
    assert trajectory.agents('nodes', 0).ids.size == 400
    for name in ('clustering', 'path_length', 'assortativity', 'modularity', 'triangle_zscore'):
        assert np.isfinite(trajectory[name][-1])
    assert trajectory['agreement'][-1] > 0.9
    assert len(np.unique(trajectory.agents('nodes', 0).values['community'])) == 4


def test_network_structure_analysis_of_a_given_graph():
    """Test that Zachary's karate club gives the NetworkX clustering and path length and at least two communities."""
    graph = nx.karate_club_graph()
    trajectory = _run(
        built := network_structure_analysis(graph, reference='club', null_samples=0),
        1,
        observables=describe('network_structure_analysis').observables_of(built),
    )
    assert trajectory['clustering'][-1] == pytest.approx(nx.average_clustering(graph), abs=1e-12)
    assert trajectory['path_length'][-1] == pytest.approx(nx.average_shortest_path_length(graph), abs=1e-12)
    assert len(np.unique(trajectory.agents('nodes', 0).values['community'])) >= 2
    assert trajectory['modularity'][-1] > 0.35
    assert trajectory['triangle_zscore'][-1] == 0.0


def test_koopman_slow_manifold_eigenfunctions_decay_exponentially():
    """Test that the Koopman eigenfunctions x1 and x2 - b x1^2 decay as exp(mu t) and exp(lam t)."""
    mu, lam = -0.05, -1.0
    trajectory = Simulation(koopman_slow_manifold(1.0, -1.0, mu=mu, lam=lam), dt=0.1).run(until=40.0).trajectory_
    times = 0.1 * np.arange(len(trajectory['flow.x1']))
    b = lam / (lam - 2 * mu)
    fast = trajectory['flow.x2'] - b * trajectory['flow.x1'] ** 2
    np.testing.assert_allclose(trajectory['flow.x1'], np.exp(mu * times), rtol=1e-5)
    np.testing.assert_allclose(fast, (-1.0 - b) * np.exp(lam * times), atol=1e-5)


def test_ou_driven_rossler_control_has_the_stationary_variance():
    """Test that the control is an Ornstein-Uhlenbeck process of variance rho^2 / (2 tau) and the flow stays bounded."""
    solver = describe('ou_driven_rossler').defaults['solver']
    system = ou_driven_rossler(tau=1.0, rho=0.2)
    trajectory = Simulation(system, dt=0.1, solver=solver, random_state=0).run(until=2000.0).trajectory_
    assert trajectory['rossler.eta'].var() == pytest.approx(0.2**2 / 2, rel=0.1)
    assert np.abs(trajectory['rossler.x']).max() < 20
