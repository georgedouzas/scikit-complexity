"""Test stochastic maps: random draws in update laws, Markov chains, and hidden Markov sources."""

import time
from typing import Annotated

import numpy as np
import pytest
import quantecon
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import EmissionLaw, RateLaw, TransitionLaw, UpdateLaw
from skcomplex.core.components.laws.utils import bernoulli, categorical, normal, shift_member, total, uniform, where
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import IncompatibleLawsError, ParameterError, ValidationError
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec

CHAIN = [[0.9, 0.1], [0.4, 0.6]]


EMISSION = [[0.7, 0.3], [0.2, 0.8]]


BINARY = FiniteEnumeratedSet([0, 1])


class UniformElement(Element):
    """Uniform element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x

    def _get_laws(self, s):
        """Return the laws of the uniform element."""
        return [UpdateLaw(s.x, uniform('u', 0, 1))]


class WalkerElement(Element):
    """Walker element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x

    def _get_laws(self, s):
        """Return the laws of the walker element."""
        return [UpdateLaw(s.x, s.x + normal('step', 0, 1))]


class FlipElement(Element):
    """Flip element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a: Annotated[object, StateSpec()] = 0.0,
        b: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a = a
        self.b = b

    def _get_laws(self, s):
        """Return the laws of the flip element."""
        return [UpdateLaw(s.a, bernoulli('flip', 0.5)), UpdateLaw(s.b, 1 - bernoulli('flip', 0.5))]


class RampElement(Element):
    """Ramp element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
        y: Annotated[object, StateSpec()] = 0.0,
        p: Annotated[object, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.p = p

    def _get_laws(self, s):
        """Return the laws of the ramp element."""
        return [UpdateLaw(s.x, s.x + 1), UpdateLaw(s.y, bernoulli('b', s.p * s.x))]


class ConflictElement(Element):
    """Conflict element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a: Annotated[object, StateSpec()] = 0.0,
        b: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a = a
        self.b = b

    def _get_laws(self, s):
        """Return the laws of the conflict element."""
        return [UpdateLaw(s.a, uniform('u', 0, 1)), UpdateLaw(s.b, uniform('u', 0, 2))]


class DriftingElement(Element):
    """Drifting element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x

    def _get_laws(self, s):
        """Return the laws of the drifting element."""
        return [RateLaw(s.x, normal('kick', 0, 1))]


class ChainElement(Element):
    """Chain element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(domain=BINARY)] = 0,
        stay: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.9,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.stay = stay

    def _get_laws(self, s):
        """Return the laws of the chain element."""
        return [TransitionLaw(s.x, [[s.stay, 1 - s.stay], [0.4, 0.6]])]


class BrokenElement(Element):
    """Broken element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(domain=BINARY)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x

    def _get_laws(self, s):
        """Return the laws of the broken element."""
        return [TransitionLaw(s.x, [[0.8, 0.1], [0.4, 0.6]])]


class HiddenElement(Element):
    """Hidden Markov source whose laws come in one of two orders."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        h: Annotated[object, StateSpec(domain=BINARY)] = 0,
        o: Annotated[object, StateSpec(domain=BINARY)] = 0,
        emission: list[list[float]] | None = None,
        order: int = 1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.h = h
        self.o = o
        self.emission = emission
        self.order = order

    def _get_laws(self, s):
        """Return the transition of the hidden state and the emission of the observed one."""
        laws = [TransitionLaw(s.h, CHAIN), EmissionLaw(s.o, s.h, EMISSION if self.emission is None else self.emission)]
        return laws if self.order == 1 else laws[::-1]


def _run(system, until, backend=None, random_state=0):
    """Return the trajectory of a system run with frames at every step."""
    return Simulation(system, dt=1.0, random_state=random_state, backend=backend).prepare().run(until).trajectory_


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
    x = _run(System(elements=[UniformElement('a')]), 100000).values('a.x')[1:]
    assert abs(x.mean() - 0.5) < 0.005
    assert abs(np.corrcoef(x[:-1], x[1:])[0, 1]) < 0.01


def test_compiler_draws_independently_for_members():
    """Test that 1000 normal walkers reach variance 100 after 100 steps with uncorrelated halves."""
    final = _run(System(elements=[GroupElement('g', element=WalkerElement, size=1000)]), 100).values('g.x')[-1]
    assert final.var() == pytest.approx(100, rel=0.15)
    assert abs(np.corrcoef(final[:500], final[500:])[0, 1]) < 0.1


def test_compiler_reads_arguments_before_the_step():
    """Test that a draw's probability uses the state and parameter values before the step."""
    y = _run(System(elements=[RampElement('r')]), 20).values('r.y')
    assert y[1] == 0
    assert np.all(y[3:] == 1)


def test_compiler_keeps_conditions_on_draws_undecided():
    """Test that a condition on a draw with constant arguments is drawn at every step, not decided when written."""

    class CoinElement(Element):
        """Coin element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the coin element."""
            return [UpdateLaw(s.x, where(bernoulli('b', 0.5) > 0, 1, 0))]

    coin = CoinElement
    x = _run(System(elements=[coin('c')]), 20000).values('c.x')[1:]
    assert np.mean(x) == pytest.approx(0.5, abs=0.02)


def test_compiler_shares_a_draw_within_a_step():
    """Test that two uses of one draw name read the same value at every frame."""
    trajectory = _run(System(elements=[FlipElement('f')]), 200)
    assert np.all((trajectory.values('f.a') + trajectory.values('f.b'))[1:] == 1)


def test_compiler_reads_the_draw_of_the_member_at_an_offset():
    """Test that a draw read at a member offset is the value the member at that offset draws."""

    class PairedElement(Element):
        """Paired element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            a: Annotated[object, StateSpec()] = 0.0,
            b: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.a = a
            self.b = b

        def _get_laws(self, s):
            """Return the laws of the paired element."""
            return [UpdateLaw(s.a, uniform('u', 0, 1)), UpdateLaw(s.b, shift_member(uniform('u', 0, 1), -1))]

    trajectory = _run(System(elements=[GroupElement('g', element=PairedElement, size=5)]), 20)
    a, b = trajectory.values('g.a')[1:], trajectory.values('g.b')[1:]
    assert np.array_equal(b, np.roll(a, 1, axis=1))
    assert len(np.unique(a)) == a.size


def test_compiler_rejects_conflicting_draws():
    """Test that one draw name with two argument lists raises an error naming both declarations."""
    declarations = r"c uses the draw name 'u' for uniform_u\(0, 1\) and for uniform_u\(0, 2\)"
    with pytest.raises(ValidationError, match=declarations):
        PassCompiler().prepare(System(elements=[ConflictElement('c')])).run()


def test_compiler_rejects_draws_in_rate_laws():
    """Test that a draw in a rate law raises an error that suggests a noise law."""
    with pytest.raises(IncompatibleLawsError, match=r'RateLaw of d holds the random draw normal_kick.*Noise laws'):
        PassCompiler().prepare(System(elements=[DriftingElement('d')])).run()


def test_transition_reaches_the_stationary_distribution():
    """Test that a two-symbol chain spends the stationary fraction of steps in its first symbol."""
    x = _codes(_run(System(elements=[ChainElement('c')]), 100000).values('c.x'))
    stationary = quantecon.MarkovChain(np.array(CHAIN)).stationary_distributions[0]
    assert np.mean(x == 0) == pytest.approx(stationary[0], abs=0.01)


def test_transition_follows_a_live_matrix_change():
    """Test that the run lengths of the first symbol fall from mean 10 to mean 2 after a live change."""
    simulation = Simulation(System(elements=[ChainElement('c')]), dt=1.0, random_state=0).prepare().run(50000)
    simulation.update_parameters({'c.stay': 0.5})
    simulation.extend(100000)
    x = _codes(simulation.trajectory_.values('c.x'))
    assert _run_lengths(x[:50000], 0).mean() == pytest.approx(10, rel=0.1)
    assert _run_lengths(x[50002:], 0).mean() == pytest.approx(2, rel=0.1)


def test_transition_rejects_a_row_that_does_not_sum_to_one():
    """Test that compiling a matrix with a row that sums to 0.9 names the law, the row, and the sum."""
    with pytest.raises(ValidationError, match=r'row 0 of the transition law of b\.x.* sums to 0\.9'):
        PassCompiler().prepare(System(elements=[BrokenElement('b')])).run()


def test_transition_rejects_a_live_change_that_breaks_a_row():
    """Test that a parameter change that makes an entry negative is rejected and the value is kept."""
    simulation = Simulation(System(elements=[ChainElement('c')]), dt=1.0, random_state=0).prepare().run(10)
    with pytest.raises(ParameterError, match=r'c\.stay.*entry \(0, 1\) of the transition law of c\.x.*below 0'):
        simulation.update_parameters({'c.stay': 1.2})
    assert simulation.problem_.parameters.read(simulation.parameters_, 'c.stay') == pytest.approx(0.9)


def test_emission_aligns_with_the_hidden_state():
    """Test that an identity emission matrix copies the hidden sequence frame by frame."""
    trajectory = _run(System(elements=[HiddenElement('s', emission=[[1, 0], [0, 1]])]), 2000)
    assert np.array_equal(_codes(trajectory.values('s.h')), _codes(trajectory.values('s.o')))


def test_emission_draws_from_the_selected_row():
    """Test that the observed frequencies given each hidden state match the emission rows."""
    trajectory = _run(System(elements=[HiddenElement('s')]), 100000)
    hidden, observed = _codes(trajectory.values('s.h'))[1:], _codes(trajectory.values('s.o'))[1:]
    for row in (0, 1):
        assert np.mean(observed[hidden == row] == 0) == pytest.approx(EMISSION[row][0], abs=0.01)


def test_compiler_reproduces_runs_with_one_seed():
    """Test that a hidden Markov source run twice with random state 0 gives equal trajectories."""
    first = _run(System(elements=[HiddenElement('s')]), 500)
    second = _run(System(elements=[HiddenElement('s')]), 500)
    assert np.array_equal(_codes(first.values('s.o')), _codes(second.values('s.o')))


def test_compiler_ignores_the_order_of_laws():
    """Test that the same source with its laws in two orders gives equal trajectories."""
    first = _run(System(elements=[HiddenElement('s', order=1)]), 500)
    second = _run(System(elements=[HiddenElement('s', order=2)]), 500)
    assert np.array_equal(_codes(first.values('s.h')), _codes(second.values('s.h')))
    assert np.array_equal(_codes(first.values('s.o')), _codes(second.values('s.o')))


def test_compiler_draws_beside_reductions():
    """Test that a draw compared with a weighted sum over a group reads both values, each in its own slot."""
    weights = np.array([[0.0, 5.0], [-5.0, 0.0]])

    class SpinsElement(Element):
        """Spins element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            s: Annotated[object, StateSpec()] = 1.0,
            h: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.s = s
            self.h = h

        def _get_laws(self, s):
            """Return the laws of the spins element."""
            return [
                UpdateLaw(s.h, where(uniform('u', 0, 1e-9) < total(s.s, weights=weights), 1, -1)),
                UpdateLaw(s.s, s.s),
            ]

    spins = SpinsElement
    trajectory = _run(System(elements=[GroupElement('g', element=spins, size=2)]), 3)
    assert trajectory.values('g.h')[-1].tolist() == [1.0, -1.0]


def test_compiler_agrees_across_backends():
    """Test that uniform, Bernoulli, and categorical draws give equal trajectories on NumPy and Numba."""

    class MixedElement(Element):
        """Mixed element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 0.0,
            k: Annotated[object, StateSpec()] = 0.0,
            b: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x
            self.k = k
            self.b = b

        def _get_laws(self, s):
            """Return the laws of the mixed element."""
            return [
                UpdateLaw(s.x, s.x / 2 + uniform('u', 0, 1)),
                UpdateLaw(s.k, categorical('k', [1, 2, s.x + 1])),
                UpdateLaw(s.b, where(s.k > 0, bernoulli('b', 0.3), 0)),
            ]

    mixed = MixedElement
    system = System(elements=[GroupElement('m', element=mixed, size=50)])
    numpy, numba = _run(system, 300, NumpyBackend()), _run(system, 300, NumbaBackend())
    for name in ('m.x', 'm.k', 'm.b'):
        assert np.allclose(numpy.values(name), numba.values(name), atol=1e-12)


def test_compiler_keeps_deterministic_maps_unchanged():
    """Test that a map without draws keeps its deterministic signature and has no draws."""

    class LogisticElement(Element):
        """Logistic element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 0.3,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the logistic element."""
            return [UpdateLaw(s.x, 3.7 * s.x * (1 - s.x))]

    logistic = LogisticElement
    problem = PassCompiler().prepare(System(elements=[logistic('l')])).run().problem_
    assert not problem.stochastic
    assert problem.update(0.0, problem.initial_state, problem.initial_parameters).tolist() == pytest.approx([0.777])


def test_compiler_writes_draws_into_equations():
    """Test that the equation of a component that draws reads the uniform value of the draw."""
    problem = PassCompiler().prepare(System(elements=[RampElement('r')])).run().problem_
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


@pytest.mark.benchmark
def test_compiler_runs_compiled_chains_fast():
    """Test that 100000 two-state chains take less than 10 times a deterministic map of the same size."""

    class TwoElement(Element):
        """Two element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec(domain=BINARY)] = 0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the two element."""
            return [TransitionLaw(s.x, CHAIN)]

    two = TwoElement

    class FlipperElement(Element):
        """Flipper element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the flipper element."""
            return [UpdateLaw(s.x, 1 - s.x)]

    flip = FlipperElement
    system = System(elements=[GroupElement('c', element=two, size=100000)])
    chains = PassCompiler(backend=NumbaBackend()).prepare(system).run().problem_
    system = System(elements=[GroupElement('c', element=flip, size=100000)])
    flips = PassCompiler(backend=NumbaBackend()).prepare(system).run().problem_
    rng = np.random.default_rng(0)
    assert _elapsed(chains, rng) < 10 * _elapsed(flips, rng)
