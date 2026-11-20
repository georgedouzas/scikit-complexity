"""Test the `_law_kernel` module of `skcomplex.core.engine.problems`."""


import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR

from skcomplex.core.components import Observable
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import CallSpec, JumpLaw, RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import categorical, divergence, reduction, total, where
from skcomplex.core.components.spaces import FiniteVolumeSpec, LatticeSpace
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.solvers import RK4Solver, SSPRKSolver
from skcomplex.core.errors import BackendWarning, ValidationError
from skcomplex.core.systems import System

STEPS = 2000
MATRIX = [[0.9, 0.1], [0.4, 0.6]]
BATCH_SHAPES: list[tuple[int, ...]] = []


def _counts(label, laws, *, n=1.0, r=0.5, size=None):
    """Return an element with one count `n` and one rate `r`, and the interaction whose `laws` change them."""
    return Element(label, values={'n': n, 'r': r}, size=size), Interaction(
        f'{label}_laws',
        roles={'target': label},
        laws=laws,
    )


def _system(label, laws, *, n=1.0, r=0.5, size=None, **settings):
    """Return the system of one element of counts whose interaction `laws` sets the laws of the namespace."""
    element, interaction = _counts(label, laws, n=n, r=r, size=size)
    return System(elements=[element], interactions=[interaction], **settings)


def _relative(s):
    """Return the growth, whose rate reads the values of its role by name, under any label."""
    return [RateLaw(s.target.n, CallSpec(lambda t, values: values['target.r'] * values['target.n'], relative=True))]


def _symbolic(s):
    """Return the symbolic form of the growth."""
    return [RateLaw(s.target.n, s.target.r * s.target.n)]


def _policy(s):
    """Return the random walk, whose update draws from the stream of the simulation."""
    return [UpdateLaw(s.target.n, CallSpec(lambda t, n, rng: n + rng.normal(), (s.target.n,), random=True))]


def _batch(s):
    """Return the growth, whose rate runs once over a whole group and counts its calls."""

    def rate(t, n, r):
        BATCH_SHAPES.append(np.shape(n))
        return r * n

    return [RateLaw(s.target.n, CallSpec(rate, (s.target.n, s.target.r), vectorized=True))]


def test_call_rejects_relative_names_next_to_arguments():
    """Test that a relative call takes no arguments."""
    with pytest.raises(ValidationError, match='remove the arguments'):
        CallSpec(lambda t, n: n, (SR(1),), relative=True)._check()


def test_relative_call_reads_its_own_values_under_every_label():
    """Test that one function reads its own states under two labels and in a nested system."""
    reference = _system('a', _symbolic).prepare().simulate(until=1.0, dt=0.01)
    expected = reference.get_values('a.n')[-1]
    c, c_laws = _counts('c', _relative)
    inner = System('inner', elements=[c], interactions=[c_laws])
    (a, a_laws), (b, b_laws) = _counts('a', _relative), _counts('b', _relative, n=1.0)
    system = System(elements=[a, b], interactions=[a_laws, b_laws], systems=[inner])
    trajectory = system.prepare().simulate(until=1.0, dt=0.01)
    for name in ('a.n', 'b.n', 'inner/c.n'):
        assert trajectory.get_values(name)[-1] == pytest.approx(expected, rel=1e-6)


def test_random_call_repeats_with_one_seed():
    """Test that a function that draws from the stream it receives repeats its run with one seed."""

    def run(seed):
        trajectory = _system('a', _policy).prepare().simulate(until=50, random_state=seed)
        return trajectory.get_values('a.n')

    first, second, other = run(1), run(1), run(2)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_vectorized_call_runs_once_over_the_group():
    """Test that a function declared to run over the whole group gets arrays of all members in one call."""
    BATCH_SHAPES.clear()
    trajectory = _system('herd', _batch, size=1000).prepare().simulate(until=0.03, dt=0.01)
    assert set(BATCH_SHAPES) == {(1000,)}
    assert np.asarray(trajectory.get_values('herd.n'))[-1] == pytest.approx(np.full(1000, np.exp(0.015)), rel=1e-4)


def _mean_field(s):
    """Return the relaxation towards the mean, a rate that runs once over a whole group for every member."""
    n = s.target.n
    return [RateLaw(n, CallSpec(lambda t, n: np.full(n.size, n.mean()) - n, (n,), vectorized=True))]


def test_vectorized_call_reads_every_member_in_the_jacobian_pattern():
    """Test that a call over the whole group may read any member, so its Jacobian pattern couples all members."""
    problem = _system('herd', _mean_field, size=4, backend=NumpyBackend()).prepare()._problem
    assert problem.jacobian.get_pattern().toarray().all()


@pytest.mark.parametrize(
    'value',
    [CallSpec(lambda t, values: values['target.r'] * values['target.n'], relative=True)],
)
def test_call_options_on_numba_run_in_python(value):
    """Test that a call with an option runs on the Numba backend with the warning of the fallback."""

    with pytest.warns(BackendWarning, match='RateLaw laws in Python'):
        problem = _system('a', lambda s: [RateLaw(s.target.n, value)], backend=NumbaBackend()).prepare()._problem
    assert problem.backends['RateLaw'] == 'python'


def _jumps(change, backend):
    """Return the system of a count that jumps at the rate `r` by a number or a function."""
    return _system('a', lambda s: [JumpLaw(s.target.r, {s.target.n: change})], backend=backend)


def test_jump_change_function_equals_the_number_on_numpy():
    """Test that a jump whose change is a function gives the run of the number that it returns."""
    system, reference = _jumps(lambda t, values: 2.0, NumpyBackend()), _jumps(2.0, NumpyBackend())
    runs = [item.prepare().simulate(until=20.0, random_state=3) for item in (system, reference)]
    assert np.array_equal(runs[0].get_values('a.n'), runs[1].get_values('a.n'))


def test_jump_change_function_reads_the_state_at_each_firing():
    """Test that each firing applies the change the function returns at the state of that firing."""
    change = CallSpec(lambda t, values: values['target.n'], relative=True)
    trajectory = _jumps(change, NumpyBackend()).prepare().simulate(until=20.0, random_state=0)
    counts = np.unique(trajectory.get_values('a.n'))
    assert set(np.log2(counts)) <= set(range(64))
    assert counts.size > 1


def test_jump_change_function_on_numba_runs_in_python():
    """Test that a jump whose change is a function falls back with the warning on the Numba backend."""
    with pytest.warns(BackendWarning, match='JumpLaw laws in Python'):
        trajectory = _jumps(lambda t, values: 2.0, NumbaBackend()).prepare().simulate(until=20.0, random_state=3)
    assert np.all(np.asarray(trajectory.get_values('a.n')) % 2 == 1)


def _step(s):
    """Return the step of a two-state chain that draws its next state from the row of `MATRIX` of its state."""
    x = s.target.x
    weights = [where(x == 0, MATRIX[0][column], MATRIX[1][column]) for column in range(2)]
    return [UpdateLaw(x, categorical('step', weights))]


def _stepper(seed):
    """Return a function that draws the next state of a two-state chain from the row of `MATRIX` of its state."""
    rng = np.random.default_rng(seed)

    def step(t, values):
        """Return the next state."""
        return float(rng.choice(2, p=MATRIX[int(values['a.x'])]))

    return step


def _chain(step, backend=None):
    """Return the states of a long run of a chain whose step is a law function or a function of the values."""
    laws = step if step is _step else lambda s: [UpdateLaw(s.target.x, step)]
    chain = Interaction('chain', roles={'target': 'a'}, laws=laws)
    system = System(elements=[Element('a', values={'x': 0})], interactions=[chain], backend=backend or NumpyBackend())
    return np.asarray(system.prepare().simulate(until=STEPS, random_state=7).get_values('a.x'), dtype=int)


def test_transition_function_matches_the_frequencies_of_the_symbolic_matrix():
    """Test that a function of the values gives the stationary frequency of the symbolic matrix."""
    symbolic, function = _chain(_step), _chain(_stepper(7))
    assert symbolic.mean() == pytest.approx(0.2, abs=0.05)
    assert function.mean() == pytest.approx(0.2, abs=0.05)


def test_transition_function_on_numba_runs_in_python():
    """Test that a function of the values falls back with the warning on the Numba backend."""
    with pytest.warns(BackendWarning, match='UpdateLaw laws in Python'):
        states = _chain(_stepper(7), NumbaBackend())
    assert states.mean() == pytest.approx(0.2, abs=0.05)


def _renewals(waiting):
    """Return the laws of a count of renewals after waiting times given as a draw or a function."""

    def laws(s):
        """Return the rate and the renewal."""
        n = s.target.n
        return [RateLaw(n, 0.0), JumpLaw(waiting=waiting, reset={n: n + 1}, label='renewal')]

    return laws


def _waiting_sampler(rng, size, t, n, r):
    """Return constant waiting times, checking the values it receives."""
    assert np.shape(t) == (size,)
    assert np.ndim(n) in {0, 1}
    return rng.uniform(r, r) * np.ones(size)


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_renewal_sampler_counts_like_its_constant_waiting_time(backend):
    """Test that a waiting time given as a function renews like the constant it returns."""
    system = _system('a', _renewals(_waiting_sampler), n=0.0, solver=RK4Solver(dt=0.01), backend=backend())
    with_backend = system.prepare().simulate(until=10.2, random_state=0)
    assert with_backend.get_values('a.n')[-1] == pytest.approx(20.0)


def test_jump_rejects_a_function_in_the_reset_of_a_renewal():
    """Test that a renewal says that its reset is symbolic."""
    with pytest.raises(ValidationError, match='takes symbolic values'):
        JumpLaw(waiting=_waiting_sampler, reset={SR.var('n'): lambda t, values: 0.0}).prepare()


def _sum(s):
    """Return the rate of a member, the sum of its group."""
    return [RateLaw(s.target.n, total(s.target.n))]


def _function_sum(s):
    """Return the rate of a member, the sum of its group through a function."""
    return [RateLaw(s.target.n, reduction(np.sum, s.target.n))]


def _reduced(laws, backend):
    """Return the final values of a group that grows by a reduction over it."""
    system = _system('herd', laws, n=np.array([1.0, 2.0, 4.0]), size=3, backend=backend)
    trajectory = system.prepare().simulate(until=0.5, dt=0.01)
    return np.asarray(trajectory.get_values('herd.n'))[-1]


def test_reduction_function_equals_the_symbolic_reduction():
    """Test that a reduction by a function gives the trajectory of the symbolic reduction."""
    assert _reduced(_function_sum, NumpyBackend()) == pytest.approx(_reduced(_sum, NumpyBackend()), rel=1e-9)


def test_reduction_function_on_numba_runs_in_python():
    """Test that a reduction by a function falls back with the warning on the Numba backend."""
    with pytest.warns(BackendWarning, match='the function reduction'):
        values = _reduced(_function_sum, NumbaBackend())
    assert values == pytest.approx(_reduced(_sum, NumpyBackend()), rel=1e-9)


def _system_output(t, values):
    """Return the value of a system trained elsewhere from the values of a system."""
    return 2.0 * values['a.n'] + t


def test_observable_function_records_its_value_at_every_frame():
    """Test that an observable given as a function gives the expression that it computes at every frame."""
    observables = [Observable('learned', function=_system_output)]
    for backend in (NumpyBackend(), NumbaBackend()):
        system = _system('a', _symbolic, backend=backend).prepare()
        trajectory = system.simulate(until=1.0, dt=0.1, observables=observables)
        expected = 2.0 * np.asarray(trajectory.get_values('a.n')) + np.asarray(trajectory.times)
        assert np.asarray(trajectory.get_values('learned')) == pytest.approx(expected)


def _burgers_speed(t, values):
    """Return the wave speed of the Burgers flux."""
    return values['u']


def _burgers(scheme):
    """Return the final values of a Burgers field under a finite-volume scheme."""
    n = 40
    centers = (np.arange(n) + 0.5) * 2 * np.pi / n
    space = LatticeSpace(n, spacing=2 * np.pi / n, discretization='finite-volume', scheme=scheme)
    field = Element('line', values={'u': 1.0 + 0.5 * np.sin(centers)}, space=space)
    flux = Interaction(
        'flux',
        roles={'target': 'line'},
        laws=lambda s: [RateLaw(s.target.u, -divergence([s.target.u**2 / 2]))],
    )
    system = System(elements=[field], interactions=[flux], solver=SSPRKSolver())
    return np.asarray(system.prepare().simulate(until=0.5, dt=0.5).get_values('line.u'))[-1]


def test_scheme_speed_function_equals_the_eigenvalues():
    """Test that a wave speed given as a function gives the run of the same speed from the Jacobian."""
    assert _burgers(FiniteVolumeSpec(speed=_burgers_speed)) == pytest.approx(_burgers(None), abs=1e-9)
