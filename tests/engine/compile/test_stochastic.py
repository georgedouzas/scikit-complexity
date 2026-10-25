"""Test stochastic maps: random draws in update laws, Markov chains, and hidden Markov sources."""

import time

import numpy as np
import pytest
import quantecon

from skcomplex.core.base import IncompatibleLawsError, NumericDomain, ParameterError, ValidationError, parameter, state
from skcomplex.core.components import ElementComponent
from skcomplex.core.laws import (
    EmissionLaw,
    RateLaw,
    TransitionLaw,
    UpdateLaw,
    bernoulli,
    categorical,
    normal,
    total,
    uniform,
    where,
)
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation

CHAIN = [[0.9, 0.1], [0.4, 0.6]]
EMISSION = [[0.7, 0.3], [0.2, 0.8]]
BINARY = NumericDomain.categorical(0, 1)

UniformComponent = ElementComponent.define(
    'UniformComponent',
    states={'x': 0.0},
    laws=lambda s: [UpdateLaw(s.x, uniform('u', 0, 1))],
)
WalkerComponent = ElementComponent.define(
    'WalkerComponent',
    states={'x': 0.0},
    laws=lambda s: [UpdateLaw(s.x, s.x + normal('step', 0, 1))],
)
FlipComponent = ElementComponent.define(
    'FlipComponent',
    states={'a': 0.0, 'b': 0.0},
    laws=lambda s: [UpdateLaw(s.a, bernoulli('flip', 0.5)), UpdateLaw(s.b, 1 - bernoulli('flip', 0.5))],
)
RampComponent = ElementComponent.define(
    'RampComponent',
    states={'x': 0.0, 'y': 0.0},
    parameters={'p': 0.5},
    laws=lambda s: [UpdateLaw(s.x, s.x + 1), UpdateLaw(s.y, bernoulli('b', s.p * s.x))],
)
ConflictComponent = ElementComponent.define(
    'ConflictComponent',
    states={'a': 0.0, 'b': 0.0},
    laws=lambda s: [UpdateLaw(s.a, uniform('u', 0, 1)), UpdateLaw(s.b, uniform('u', 0, 2))],
)
DriftingComponent = ElementComponent.define(
    'DriftingComponent',
    states={'x': 0.0},
    laws=lambda s: [RateLaw(s.x, normal('kick', 0, 1))],
)
ChainComponent = ElementComponent.define(
    'ChainComponent',
    states={'x': state(default=0, domain=BINARY)},
    parameters={'stay': parameter(0.9, domain=NumericDomain.NONNEGATIVE)},
    laws=lambda s: [TransitionLaw(s.x, [[s.stay, 1 - s.stay], [0.4, 0.6]])],
)
BrokenComponent = ElementComponent.define(
    'BrokenComponent',
    states={'x': state(default=0, domain=BINARY)},
    laws=lambda s: [TransitionLaw(s.x, [[0.8, 0.1], [0.4, 0.6]])],
)


def _hidden(emission, order=1):
    """Return a hidden Markov source element type with laws in one of two orders."""

    def laws(s):
        found = [TransitionLaw(s.h, CHAIN), EmissionLaw(s.o, s.h, emission)]
        return found if order == 1 else found[::-1]

    return ElementComponent.define(
        'HiddenComponent',
        states={'h': state(default=0, domain=BINARY), 'o': state(default=0, domain=BINARY)},
        laws=laws,
    )


def _run(system, until, backend='numpy', random_state=0):
    """Return the trajectory of a system run with frames at every step."""
    return Simulation(system, dt=1.0, random_state=random_state, backend=backend).run(until).trajectory_


def _codes(values):
    """Return the stored indices of a categorical series given as labels or indices."""
    return np.asarray(values, dtype=float)


def _run_lengths(series, symbol):
    """Return the lengths of the runs of one symbol in a series."""
    padded = np.concatenate([[0], (series == symbol).astype(int), [0]])
    edges = np.flatnonzero(np.diff(padded))
    return edges[1::2] - edges[::2]


def test_compile_system_draws_uniform_values():
    """Test that a uniform draw has mean 0.5 and no lag-1 autocorrelation over 100000 steps."""
    x = _run(System(components=[UniformComponent('a')]), 100000)['a.x'][1:]
    assert abs(x.mean() - 0.5) < 0.005
    assert abs(np.corrcoef(x[:-1], x[1:])[0, 1]) < 0.01


def test_compile_system_draws_independently_for_members():
    """Test that 1000 normal walkers reach variance 100 after 100 steps with uncorrelated halves."""
    final = _run(System(components=[WalkerComponent.group('g', 1000)]), 100)['g.x'][-1]
    assert final.var() == pytest.approx(100, rel=0.15)
    assert abs(np.corrcoef(final[:500], final[500:])[0, 1]) < 0.1


def test_compile_system_reads_arguments_before_the_step():
    """Test that a draw's probability uses the state and parameter values before the step."""
    y = _run(System(components=[RampComponent('r')]), 20)['r.y']
    assert y[1] == 0
    assert np.all(y[3:] == 1)


def test_compile_system_keeps_conditions_on_draws_undecided():
    """Test that a condition on a draw with constant arguments is drawn at every step, not decided when written."""
    coin = ElementComponent.define(
        'CoinComponent',
        states={'x': 0.0},
        laws=lambda s: [UpdateLaw(s.x, where(bernoulli('b', 0.5) > 0, 1, 0))],
    )
    x = _run(System(components=[coin('c')]), 20000)['c.x'][1:]
    assert np.mean(x) == pytest.approx(0.5, abs=0.02)


def test_compile_system_shares_a_draw_within_a_step():
    """Test that two uses of one draw name read the same value at every frame."""
    trajectory = _run(System(components=[FlipComponent('f')]), 200)
    assert np.all((trajectory['f.a'] + trajectory['f.b'])[1:] == 1)


def test_compile_system_rejects_conflicting_draws():
    """Test that one draw name with two argument lists raises an error naming both declarations."""
    declarations = r"c uses the draw name 'u' for uniform_u\(0, 1\) and for uniform_u\(0, 2\)"
    with pytest.raises(ValidationError, match=declarations):
        compile_system(System(components=[ConflictComponent('c')]))


def test_compile_system_rejects_draws_in_rate_laws():
    """Test that a draw in a rate law raises an error that suggests a noise law."""
    with pytest.raises(IncompatibleLawsError, match=r'RateLaw of d holds the random draw normal_kick.*Noise laws'):
        compile_system(System(components=[DriftingComponent('d')]))


def test_transition_reaches_the_stationary_distribution():
    """Test that a two-symbol chain spends the stationary fraction of steps in its first symbol."""
    x = _codes(_run(System(components=[ChainComponent('c')]), 100000)['c.x'])
    stationary = quantecon.MarkovChain(np.array(CHAIN)).stationary_distributions[0]
    assert np.mean(x == 0) == pytest.approx(stationary[0], abs=0.01)


def test_transition_follows_a_live_matrix_change():
    """Test that the run lengths of the first symbol fall from mean 10 to mean 2 after a live change."""
    simulation = Simulation(System(components=[ChainComponent('c')]), dt=1.0, random_state=0).run(50000)
    simulation.update_parameters({'c.stay': 0.5})
    simulation.extend(100000)
    x = _codes(simulation.trajectory_['c.x'])
    assert _run_lengths(x[:50000], 0).mean() == pytest.approx(10, rel=0.1)
    assert _run_lengths(x[50002:], 0).mean() == pytest.approx(2, rel=0.1)


def test_transition_rejects_a_row_that_does_not_sum_to_one():
    """Test that compiling a matrix with a row that sums to 0.9 names the law, the row, and the sum."""
    with pytest.raises(ValidationError, match=r'row 0 of the transition law of b\.x.* sums to 0\.9'):
        compile_system(System(components=[BrokenComponent('b')]))


def test_transition_rejects_a_live_change_that_breaks_a_row():
    """Test that a parameter change that makes an entry negative is rejected and the value is kept."""
    simulation = Simulation(System(components=[ChainComponent('c')]), dt=1.0, random_state=0).run(10)
    with pytest.raises(ParameterError, match=r'c\.stay.*entry \(0, 1\) of the transition law of c\.x.*below 0'):
        simulation.update_parameters({'c.stay': 1.2})
    assert simulation.problem_.parameters.read(simulation.parameters_, 'c.stay') == pytest.approx(0.9)


def test_emission_aligns_with_the_hidden_state():
    """Test that an identity emission matrix copies the hidden sequence frame by frame."""
    trajectory = _run(System(components=[_hidden([[1, 0], [0, 1]])('s')]), 2000)
    assert np.array_equal(_codes(trajectory['s.h']), _codes(trajectory['s.o']))


def test_emission_draws_from_the_selected_row():
    """Test that the observed frequencies given each hidden state match the emission rows."""
    trajectory = _run(System(components=[_hidden(EMISSION)('s')]), 100000)
    hidden, observed = _codes(trajectory['s.h'])[1:], _codes(trajectory['s.o'])[1:]
    for row in (0, 1):
        assert np.mean(observed[hidden == row] == 0) == pytest.approx(EMISSION[row][0], abs=0.01)


def test_compile_system_reproduces_runs_with_one_seed():
    """Test that a hidden Markov source run twice with random state 0 gives equal trajectories."""
    first = _run(System(components=[_hidden(EMISSION)('s')]), 500)
    second = _run(System(components=[_hidden(EMISSION)('s')]), 500)
    assert np.array_equal(_codes(first['s.o']), _codes(second['s.o']))


def test_compile_system_ignores_the_order_of_laws():
    """Test that the same source with its laws in two orders gives equal trajectories."""
    first = _run(System(components=[_hidden(EMISSION, order=1)('s')]), 500)
    second = _run(System(components=[_hidden(EMISSION, order=2)('s')]), 500)
    assert np.array_equal(_codes(first['s.h']), _codes(second['s.h']))
    assert np.array_equal(_codes(first['s.o']), _codes(second['s.o']))


def test_compile_system_draws_beside_reductions():
    """Test that a draw compared with a weighted sum over a group reads both values, each in its own slot."""
    weights = np.array([[0.0, 5.0], [-5.0, 0.0]])
    spins = ElementComponent.define(
        'SpinsComponent',
        states={'s': 1.0, 'h': 0.0},
        laws=lambda s: [
            UpdateLaw(s.h, where(uniform('u', 0, 1e-9) < total(s.s, weights=weights), 1, -1)),
            UpdateLaw(s.s, s.s),
        ],
    )
    trajectory = _run(System(components=[spins.group('g', 2)]), 3)
    assert trajectory['g.h'][-1].tolist() == [1.0, -1.0]


def test_compile_system_agrees_across_backends():
    """Test that uniform, Bernoulli, and categorical draws give equal trajectories on NumPy and Numba."""
    mixed = ElementComponent.define(
        'MixedComponent',
        states={'x': 0.0, 'k': 0.0, 'b': 0.0},
        laws=lambda s: [
            UpdateLaw(s.x, s.x / 2 + uniform('u', 0, 1)),
            UpdateLaw(s.k, categorical('k', [1, 2, s.x + 1])),
            UpdateLaw(s.b, where(s.k > 0, bernoulli('b', 0.3), 0)),
        ],
    )
    system = System(components=[mixed.group('m', 50)])
    numpy, numba = _run(system, 300, 'numpy'), _run(system, 300, 'numba')
    for name in ('m.x', 'm.k', 'm.b'):
        assert np.allclose(numpy[name], numba[name], atol=1e-12)


def test_compile_system_keeps_deterministic_maps_unchanged():
    """Test that a map without draws keeps its deterministic signature and has no draws."""
    logistic = ElementComponent.define(
        'LogisticComponent',
        states={'x': 0.3},
        laws=lambda s: [UpdateLaw(s.x, 3.7 * s.x * (1 - s.x))],
    )
    problem = compile_system(System(components=[logistic('l')]))
    assert not problem.stochastic
    assert problem.update(0.0, problem.initial_state, problem.initial_parameters).tolist() == pytest.approx([0.777])


def test_compile_system_writes_draws_into_equations():
    """Test that the equation of a component that draws reads the uniform value of the draw."""
    problem = compile_system(System(components=[RampComponent('r')]))
    assert str(problem.equations['r.y']) == 'where(bernoulli_b_r < p_r*x_r, 1, 0)'


def _elapsed(problem, rng):
    """Return the shortest of five times a compiled discrete loop takes for 1000 steps after a warm-up step."""
    arguments = (rng,) if problem.stochastic else ()
    problem.accelerated.iterate(0.0, problem.initial_state, problem.initial_parameters, 1.0, *arguments)
    elapsed = []
    for _ in range(5):
        start = time.perf_counter()
        problem.accelerated.iterate(0.0, problem.initial_state, problem.initial_parameters, 1000.0, *arguments)
        elapsed.append(time.perf_counter() - start)
    return min(elapsed)


def test_compile_system_runs_compiled_chains_fast():
    """Test that 100000 two-state chains take less than 10 times a deterministic map of the same size."""
    two = ElementComponent.define(
        'TwoComponent',
        states={'x': state(default=0, domain=BINARY)},
        laws=lambda s: [TransitionLaw(s.x, CHAIN)],
    )
    flip = ElementComponent.define('FlipperComponent', states={'x': 0.0}, laws=lambda s: [UpdateLaw(s.x, 1 - s.x)])
    chains = compile_system(System(components=[two.group('c', 100000)]), backend='numba')
    flips = compile_system(System(components=[flip.group('c', 100000)]), backend='numba')
    rng = np.random.default_rng(0)
    assert _elapsed(chains, rng) < 10 * _elapsed(flips, rng)
