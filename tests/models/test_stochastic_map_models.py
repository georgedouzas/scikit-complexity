"""Test the models built on stochastic maps: walks, Markov sources, the chaos game, and stochastic networks."""

import numpy as np
import pytest
from scipy import integrate

from skcomplex.core.analysis import box_counting_dimension, entropy_rate
from skcomplex.core.base import ValidationError
from skcomplex.core.runtime import Simulation
from skcomplex.models import create

TWO_STATE = ((0.9, 0.1), (0.4, 0.6))


def _final(name, until, **arguments):
    """Return the simulation of a model run to `until` with frames at the start and the end only."""
    return Simulation(create(name, **arguments), dt=until, random_state=0).run(until)


def _series(name, until, **arguments):
    """Return the trajectory of a model run with frames at every step."""
    return Simulation(create(name, **arguments), dt=1, random_state=0).run(until).trajectory_


def _memory_msd(t, alpha, sigma):
    """Return the exact mean squared displacement of the random walk with memory after `t` steps from rest."""
    xx = xv = vv = 0.0
    for _ in range(t):
        xx, xv, vv = xx + 2 * xv + vv, alpha * (xv + vv), alpha**2 * vv + sigma**2
    return xx


def _hopfield_map(m, connectivity, n_patterns, beta):
    """Return the overlap after one step of the annealed approximation, Eq. (6.55) of Bertin."""
    spread = np.sqrt(connectivity * (n_patterns - m**2))

    def integrand(z):
        return np.exp(-(z**2) / 2) * np.tanh(beta * (connectivity * m + spread * z)) / np.sqrt(2 * np.pi)

    return integrate.quad(integrand, -10, 10)[0]


def test_persistent_random_walk_diffuses_faster():
    """Test that persistent walkers have a mean squared displacement per step of `(1 + rho) / (1 - rho) = 4`."""
    positions = _final('persistent_random_walk', 1000, n_walkers=2000, persistence=0.8).trajectory_['walkers.x'][-1]
    assert np.mean(positions**2) / 1000 == pytest.approx(4.0, rel=0.1)


def test_random_walk_with_memory_follows_the_exact_displacement():
    """Test that the mean squared displacement matches the exact second moments of the walk."""
    positions = _final('random_walk_with_memory', 500, n_walkers=4000).trajectory_['walkers.x'][-1]
    assert np.mean(positions**2) == pytest.approx(_memory_msd(500, 0.5, 1.0), rel=0.1)


def test_markov_source_has_the_entropy_rate_of_its_chain():
    """Test that the entropy rate estimated from block entropies is 0.569 bits."""
    symbols = _series('markov_source', 100000, transition=TWO_STATE)['source.symbol']
    assert entropy_rate(symbols.astype(int), 8) == pytest.approx(0.569, abs=0.01)


def test_hidden_markov_source_emits_at_the_stationary_frequency():
    """Test that the first symbol has the frequency `0.8 x 0.7 + 0.2 x 0.2 = 0.6`."""
    symbols = _series('hidden_markov_source', 100000)['source.symbol']
    assert np.mean(symbols == 0) == pytest.approx(0.6, abs=0.01)


def test_chaos_game_draws_the_sierpinski_triangle():
    """Test that the box-counting dimension of the final points is `log 3 / log 2` within 0.05."""
    points = _final('chaos_game', 30, n_points=100000).trajectory_['points.p'][-1]
    cells = np.clip((points * 128).astype(int), 0, 127)
    grid = np.zeros((128, 128), dtype=bool)
    grid[cells[:, 1], cells[:, 0]] = True
    assert box_counting_dimension(grid, [1, 2, 4, 8, 16, 32]) == pytest.approx(np.log(3) / np.log(2), abs=0.05)


def test_chaos_game_agrees_across_backends():
    """Test that the NumPy and Numba backends give equal trajectories of the chaos game from one seed."""
    runs = [
        Simulation(create('chaos_game', n_points=20), dt=1, random_state=0, backend=backend).run(1000).trajectory_
        for backend in ('numpy', 'numba')
    ]
    assert np.allclose(runs[0]['points.p'], runs[1]['points.p'], atol=1e-12)


def test_chaos_game_rejects_unknown_presets():
    """Test that an unknown preset raises an error that names the presets."""
    with pytest.raises(ValidationError, match='sierpinski, barnsley'):
        create('chaos_game', preset='dragon')


def test_probabilistic_boolean_network_draws_functions_with_their_probabilities():
    """Test that a node with two functions of probabilities 0.7 and 0.3 uses the first in 70 percent of the steps."""
    network = {
        'x1': (((0, 1, 1, 1, 0, 1, 1, 1), 0.7), ((0, 0, 0, 1, 0, 1, 1, 1), 0.3)),
        'x2': (((0, 1, 1, 0, 0, 1, 0, 1), 1.0),),
        'x3': (((0, 0, 0, 1, 0, 1, 1, 1), 0.5), ((0, 0, 0, 0, 0, 0, 0, 1), 0.5)),
    }
    used = _series('probabilistic_boolean_network', 20000, network=network)['x1.function'][1:]
    error = np.sqrt(0.7 * 0.3 / used.size)
    assert abs(np.mean(used == 0) - 0.7) < 3 * error


def test_intrinsic_instability_leaves_its_fixed_point():
    """Test that the resident population stays near 1 until the invader grows and then declines."""
    trajectory = _series('intrinsic_instability', 4000)
    x, y = trajectory['populations.x'], trajectory['populations.y']
    seeded = int(np.argmax(y > 0))
    assert 0 < seeded < 3000
    assert np.all(np.abs(x[100:seeded] - 1) < 0.15)
    assert x[-1] < 0.5
    assert y[-1] > 1


def test_diluted_hopfield_reaches_the_annealed_fixed_point():
    """Test that the overlap settles at the fixed point of the annealed map of Bertin within 0.05."""
    overlap = _series('diluted_hopfield', 20)['overlap']
    m = 0.3
    for _ in range(50):
        m = _hopfield_map(m, 50.0, 10, 0.06)
    assert np.mean(overlap[-5:]) == pytest.approx(m, abs=0.05)
