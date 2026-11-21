"""Test stochastic maps: random draws in update laws, Markov chains, and hidden Markov sources."""

import time
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import quantecon

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import bernoulli, categorical, normal, shift_member, total, uniform, where
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.errors import IncompatibleLawsError, ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role

CHAIN = [[0.9, 0.1], [0.4, 0.6]]
EMISSION = [[0.7, 0.3], [0.2, 0.8]]


def _system(label, values, laws, *, size=None, backend='auto'):
    """Return one element, or a group of a size, whose rule returns laws of its namespace."""
    rule = Interaction('rule', roles={'target': label}, laws=lambda s: laws(s.target))
    return System(elements=[Element(label, values=values, size=size)], interactions=[rule], backend=backend)


def _uniform(label='a'):
    """Return an element that draws a uniform value at every step."""
    return _system(label, {'x': 0.0}, lambda e: [UpdateLaw(e.x, uniform('u', 0, 1))])


def _ramp():
    """Return an element that counts up and draws a Bernoulli value of a chance that grows with the count."""
    return _system(
        'r',
        {'x': 0.0, 'y': 0.0, 'p': 0.5},
        lambda e: [UpdateLaw(e.x, e.x + 1), UpdateLaw(e.y, bernoulli('b', e.p * e.x))],
    )


def _row(name, matrix, value):
    """Return the index drawn from the row of a matrix that a value selects."""
    weights = [
        sum(where(value == row, matrix[row][column], 0) for row in range(len(matrix)))
        for column in range(len(matrix[0]))
    ]
    return categorical(name, weights)


class ChainInteraction(Interaction):
    """Markov chain of two symbols whose first symbol stays with the chance `stay`."""

    def __init__(self, label: str, *, target: Role = 'c', stay: float = 0.9) -> None:
        super().__init__(label)
        self.target = target
        self.stay = stay

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the draw of the next symbol."""
        super()._prepare(context)
        x = context.target.x
        self.laws_ = [UpdateLaw(x, _row('next', [[context.stay, 1 - context.stay], [0.4, 0.6]], x))]


def _chain():
    """Return a Markov chain of two symbols."""
    return System(elements=[Element('c', values={'x': 0})], interactions=[ChainInteraction('chain')])


class HiddenInteraction(Interaction):
    """Hidden Markov source whose laws come in one of two orders."""

    def __init__(self, label: str, *, target: Role = 's', emission: list[list[float]] | None = None, order: int = 1):
        super().__init__(label)
        self.target = target
        self.emission = emission
        self.order = order

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the transition of the hidden state and the emission of the observed one from the next hidden state."""
        super()._prepare(context)
        s = context.target
        emission = EMISSION if self.emission is None else self.emission
        following = _row('h', CHAIN, s.h)
        laws = [UpdateLaw(s.h, following), UpdateLaw(s.o, _row('o', emission, following))]
        self.laws_ = laws if self.order == 1 else laws[::-1]


def _hidden(**settings):
    """Return a hidden Markov source."""
    return System(
        elements=[Element('s', values={'h': 0, 'o': 0})],
        interactions=[HiddenInteraction('source', **settings)],
    )


class KeptRegulationInteraction(Interaction):
    """Increment of the edge or the focal member of every hyperedge by its last member, kept with a drawn chance."""

    def __init__(self, label: str, *, among: Role = 'nodes', along: str = 'hyperedges', part: str = 'edge') -> None:
        super().__init__(label, among=among, along=along)
        self.part = part

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the increment of the edge or of the focal member."""
        super()._prepare(context)
        s = context
        target = s.edge.x if self.part == 'edge' else s.member.x
        self.laws_ = [UpdateLaw(target, target + bernoulli('kept', 0.5) * s.others[-1].x, increment=True)]


def _run(system, until, backend=None, random_state=0, **settings):
    """Return the trajectory of a system run with frames at every step."""
    if backend is not None:
        system.set_params(backend=backend)
    return system.prepare().simulate(until=until, dt=1.0, random_state=random_state, **settings)


def _codes(values):
    """Return the stored indices of a categorical series given as labels or indices."""
    return np.asarray(values, dtype=float)


def _run_lengths(series, symbol):
    """Return the lengths of the runs of one symbol in a series."""
    padded = np.concatenate([[0], (series == symbol).astype(int), [0]])
    edges = np.flatnonzero(np.diff(padded))
    return edges[1::2] - edges[::2]


def test_compiler_draws_uniform_values():
    """Test that a uniform draw has mean 0.5 and no lag-1 autocorrelation over 100000 steps."""
    x = _run(_uniform(), 100000).get_values('a.x')[1:]
    assert abs(x.mean() - 0.5) < 0.005
    assert abs(np.corrcoef(x[:-1], x[1:])[0, 1]) < 0.01


def test_compiler_draws_independently_for_members():
    """Test that 1000 normal walkers reach variance 100 after 100 steps with uncorrelated halves."""
    system = _system('g', {'x': 0.0}, lambda e: [UpdateLaw(e.x, e.x + normal('step', 0, 1))], size=1000)
    final = _run(system, 100).get_values('g.x')[-1]
    assert final.var() == pytest.approx(100, rel=0.15)
    assert abs(np.corrcoef(final[:500], final[500:])[0, 1]) < 0.1


def test_compiler_reads_arguments_before_the_step():
    """Test that a draw's probability uses the state and parameter values before the step."""
    y = _run(_ramp(), 20).get_values('r.y')
    assert y[1] == 0
    assert np.all(y[3:] == 1)


def test_compiler_keeps_conditions_on_draws_undecided():
    """Test that a condition on a draw with constant arguments is drawn at every step, not decided when written."""
    system = _system('c', {'x': 0.0}, lambda e: [UpdateLaw(e.x, where(bernoulli('b', 0.5) > 0, 1, 0))])
    x = _run(system, 20000).get_values('c.x')[1:]
    assert np.mean(x) == pytest.approx(0.5, abs=0.02)


def test_compiler_shares_a_draw_within_a_step():
    """Test that two uses of one draw name read the same value at every frame."""

    def laws(e):
        return [UpdateLaw(e.a, bernoulli('flip', 0.5)), UpdateLaw(e.b, 1 - bernoulli('flip', 0.5))]

    trajectory = _run(_system('f', {'a': 0.0, 'b': 0.0}, laws), 200)
    assert np.all((trajectory.get_values('f.a') + trajectory.get_values('f.b'))[1:] == 1)


def test_compiler_reads_the_draw_of_the_member_at_an_offset():
    """Test that a draw read at a member offset is the value the member at that offset draws."""

    def laws(e):
        return [UpdateLaw(e.a, uniform('u', 0, 1)), UpdateLaw(e.b, shift_member(uniform('u', 0, 1), -1))]

    trajectory = _run(_system('g', {'a': 0.0, 'b': 0.0}, laws, size=5), 20)
    a, b = trajectory.get_values('g.a')[1:], trajectory.get_values('g.b')[1:]
    assert np.array_equal(b, np.roll(a, 1, axis=1))
    assert len(np.unique(a)) == a.size


def test_compiler_rejects_conflicting_draws():
    """Test that one draw name with two argument lists raises an error naming both declarations."""
    system = _system(
        'c',
        {'a': 0.0, 'b': 0.0},
        lambda e: [UpdateLaw(e.a, uniform('u', 0, 1)), UpdateLaw(e.b, uniform('u', 0, 2))],
    )
    declarations = r"rule uses the draw name 'u' for uniform_u\(0, 1\) and for uniform_u\(0, 2\)"
    with pytest.raises(ValidationError, match=declarations):
        system.prepare()


def test_compiler_rejects_draws_in_rate_laws():
    """Test that a draw in a rate law raises an error that suggests a noise law."""
    system = _system('d', {'x': 0.0}, lambda e: [RateLaw(e.x, normal('kick', 0, 1))])
    with pytest.raises(IncompatibleLawsError, match=r'RateLaw of rule holds the random draw normal_kick.*Noise laws'):
        system.prepare()


def test_transition_reaches_the_stationary_distribution():
    """Test that a two-symbol chain spends the stationary fraction of steps in its first symbol."""
    x = _codes(_run(_chain(), 100000).get_values('c.x'))
    stationary = quantecon.MarkovChain(np.array(CHAIN)).stationary_distributions[0]
    assert np.mean(x == 0) == pytest.approx(stationary[0], abs=0.01)


def test_transition_follows_a_live_matrix_change():
    """Test that the run lengths of the first symbol fall from mean 10 to mean 2 after a live change."""
    schedule = {'chain.stay': lambda t: 0.5 if t >= 50000 else 0.9}
    x = _codes(_run(_chain(), 150000, schedule=schedule).get_values('c.x'))
    assert _run_lengths(x[:50000], 0).mean() == pytest.approx(10, rel=0.1)
    assert _run_lengths(x[50002:], 0).mean() == pytest.approx(2, rel=0.1)


def test_emission_aligns_with_the_hidden_state():
    """Test that an identity emission matrix copies the hidden sequence frame by frame."""
    trajectory = _run(_hidden(emission=[[1, 0], [0, 1]]), 2000)
    assert np.array_equal(_codes(trajectory.get_values('s.h')), _codes(trajectory.get_values('s.o')))


def test_emission_draws_from_the_selected_row():
    """Test that the observed frequencies given each hidden state match the emission rows."""
    trajectory = _run(_hidden(), 100000)
    hidden, observed = _codes(trajectory.get_values('s.h'))[1:], _codes(trajectory.get_values('s.o'))[1:]
    for row in (0, 1):
        assert np.mean(observed[hidden == row] == 0) == pytest.approx(EMISSION[row][0], abs=0.01)


def test_compiler_reproduces_runs_with_one_seed():
    """Test that a hidden Markov source run twice with random state 0 gives equal trajectories."""
    first = _run(_hidden(), 500)
    second = _run(_hidden(), 500)
    assert np.array_equal(_codes(first.get_values('s.o')), _codes(second.get_values('s.o')))


def test_compiler_ignores_the_order_of_laws():
    """Test that the same source with its laws in two orders gives equal trajectories."""
    first = _run(_hidden(order=1), 500)
    second = _run(_hidden(order=2), 500)
    assert np.array_equal(_codes(first.get_values('s.h')), _codes(second.get_values('s.h')))
    assert np.array_equal(_codes(first.get_values('s.o')), _codes(second.get_values('s.o')))


def test_compiler_draws_beside_reductions():
    """Test that a draw compared with a weighted sum over a group reads both values, each in its own slot."""
    weights = np.array([[0.0, 5.0], [-5.0, 0.0]])

    def laws(e):
        return [UpdateLaw(e.h, where(uniform('u', 0, 1e-9) < total(e.s, weights=weights), 1, -1)), UpdateLaw(e.s, e.s)]

    trajectory = _run(_system('g', {'s': 1.0, 'h': 0.0}, laws, size=2), 3)
    assert trajectory.get_values('g.h')[-1].tolist() == [1.0, -1.0]


def _mixed(backend):
    """Return a group whose members draw uniform, categorical, and Bernoulli values."""

    def laws(e):
        return [
            UpdateLaw(e.x, e.x / 2 + uniform('u', 0, 1)),
            UpdateLaw(e.k, categorical('k', [1, 2, e.x + 1])),
            UpdateLaw(e.b, where(e.k > 0, bernoulli('b', 0.3), 0)),
        ]

    return _system('m', {'x': 0.0, 'k': 0.0, 'b': 0.0}, laws, size=50, backend=backend)


def test_compiler_agrees_across_backends():
    """Test that uniform, Bernoulli, and categorical draws give equal trajectories on NumPy and Numba."""
    numpy, numba = _run(_mixed(NumpyBackend()), 300), _run(_mixed(NumbaBackend()), 300)
    for name in ('m.x', 'm.k', 'm.b'):
        assert np.allclose(numpy.get_values(name), numba.get_values(name), atol=1e-12)


@pytest.mark.parametrize('part', ['edge', 'member'])
def test_compiler_agrees_across_backends_on_draws_of_hyperedges_of_several_sizes(part):
    """Test that a drawn increment over hyperedges of two sizes gives equal trajectories on NumPy and Numba."""
    hyperedges = [(0, 1, 4), (1, 2, 5), (2, 3, 0), (3, 4, 1), (4, 5, 0, 2), (5, 0, 1, 3)]
    space = NetworkSpace(nx.cycle_graph(6), hyperedges=hyperedges)

    def system(backend):
        return System(
            elements=[
                Element('nodes', values={'x': np.arange(1.0, 7.0)}, space=space),
                Element('links', values={'x': 0.0}, space=space, on='edges'),
            ],
            interactions=[KeptRegulationInteraction('regulation', part=part)],
            backend=backend,
        )

    numpy, numba = _run(system(NumpyBackend()), 10), _run(system(NumbaBackend()), 10)
    for name in ('nodes.x', 'links.x'):
        assert np.allclose(numpy.get_values(name), numba.get_values(name), atol=1e-12)


def test_compiler_keeps_deterministic_maps_unchanged():
    """Test that a map without draws keeps its deterministic signature and has no draws."""
    problem = _system('l', {'x': 0.3}, lambda e: [UpdateLaw(e.x, 3.7 * e.x * (1 - e.x))]).prepare()._problem
    assert not problem.stochastic
    assert problem.update(0.0, problem.initial_state, problem.initial_parameters).tolist() == pytest.approx([0.777])


def test_compiler_writes_draws_into_equations():
    """Test that the equation of a component that draws reads the uniform value of the draw."""
    problem = _ramp().prepare()._problem
    assert str(problem.equations['r.y']) == 'where(bernoulli_b_rule < p_r*x_r, 1, 0)'


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


@pytest.mark.benchmark
def test_compiler_runs_compiled_chains_fast():
    """Test that 100000 two-state chains take less than 10 times a deterministic map of the same size."""
    chains = _system(
        'c',
        {'x': 0},
        lambda e: [UpdateLaw(e.x, _row('next', CHAIN, e.x))],
        size=100000,
        backend=NumbaBackend(),
    )
    flips = _system('c', {'x': 0.0}, lambda e: [UpdateLaw(e.x, 1 - e.x)], size=100000, backend=NumbaBackend())
    rng = np.random.default_rng(0)
    assert _elapsed(chains.prepare()._problem, rng) < 10 * _elapsed(flips.prepare()._problem, rng)


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()])
def test_compiler_draws_relaxed_draws_exactly(backend):
    """Test that relaxed Bernoulli and categorical draws give exact values off JAX, as unrelaxed draws do."""

    def relaxed_laws(e):
        return [
            UpdateLaw(e.b, bernoulli('b', 0.3, temperature=0.5)),
            UpdateLaw(e.k, categorical('k', [1, 2, 3], temperature=0.5, hard=True)),
        ]

    def exact_laws(e):
        return [UpdateLaw(e.b, bernoulli('b', 0.3)), UpdateLaw(e.k, categorical('k', [1, 2, 3]))]

    values = {'b': 0.0, 'k': 0.0}
    relaxed = _run(_system('m', values, relaxed_laws, size=40, backend=backend), 50)
    exact = _run(_system('m', values, exact_laws, size=40, backend=backend), 50)
    assert set(np.unique(relaxed.get_values('m.b'))) == {0.0, 1.0}
    assert set(np.unique(relaxed.get_values('m.k'))) == {0.0, 1.0, 2.0}
    for name in ('m.b', 'm.k'):
        np.testing.assert_array_equal(relaxed.get_values(name), exact.get_values(name))
