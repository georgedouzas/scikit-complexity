"""Test the functions that the `_law_kernel` module of `skcomplex.core.engine.problems` calls."""

from typing import Annotated, ClassVar

import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR, ZZ

from skcomplex.core.components.elements import Element, FieldElement, GroupElement
from skcomplex.core.components.laws import FluxLaw, JumpLaw, RateLaw, TransitionLaw, UpdateLaw
from skcomplex.core.components.laws.utils import reduction, total
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver, SSPRKSolver
from skcomplex.core.errors import BackendWarning, ValidationError
from skcomplex.core.specs import CallSpec, ObservableSpec, ParameterSpec, StateSpec, VisualSpec

STEPS = 2000


class BaseCountElement(Element):
    """Element with one count and one rate, whose laws its subclasses return."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[object, StateSpec()] = 1.0,
        r: Annotated[object, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n
        self.r = r


class RelativeElement(BaseCountElement):
    """Element whose rate reads its own values by name, under any label."""

    def _get_laws(self, s):
        """Return the growth."""
        return [RateLaw(s.n, CallSpec(lambda t, values: values['r'] * values['n'], relative=True))]


class SymbolicElement(BaseCountElement):
    """Element with the symbolic form of the growth."""

    def _get_laws(self, s):
        """Return the growth."""
        return [RateLaw(s.n, s.r * s.n)]


class PolicyElement(BaseCountElement):
    """Element whose update draws from the stream of the simulation."""

    def _get_laws(self, s):
        """Return the random walk."""
        return [UpdateLaw(s.n, CallSpec(lambda t, n, rng: n + rng.normal(), (s.n,), random=True))]


class BatchElement(BaseCountElement):
    """Element whose rate runs once over a whole group and counts its calls."""

    shapes: ClassVar[list[tuple[int, ...]]] = []

    def _get_laws(self, s):
        """Return the growth."""

        def rate(t, n, r):
            type(self).shapes.append(np.shape(n))
            return r * n

        return [RateLaw(s.n, CallSpec(rate, (s.n, s.r), vectorized=True))]


def test_call_rejects_relative_names_next_to_arguments():
    """Test that a relative call takes no arguments."""
    with pytest.raises(ValidationError, match='remove the arguments'):
        CallSpec(lambda t, n: n, (SR(1),), relative=True).prepare()


def test_relative_call_reads_its_own_values_under_every_label():
    """Test that one function reads its own states under two labels and in a nested system."""
    reference = simulate(System(elements=[SymbolicElement('a')]), until=1.0, dt=0.01)
    expected = reference.values('a.n')[-1]
    inner = System('inner', elements=[RelativeElement('c')])
    system = System(elements=[RelativeElement('a'), RelativeElement('b', n=1.0)], systems=[inner])
    trajectory = simulate(system, until=1.0, dt=0.01)
    for name in ('a.n', 'b.n', 'inner/c.n'):
        assert trajectory.values(name)[-1] == pytest.approx(expected, rel=1e-6)


def test_random_call_repeats_with_one_seed():
    """Test that a function that draws from the stream it receives repeats its run with one seed."""

    def run(seed):
        system = System(elements=[PolicyElement('a')], time=DiscreteTime())
        trajectory = simulate(system, until=50, random_state=seed)
        return trajectory.values('a.n')

    first, second, other = run(1), run(1), run(2)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_vectorized_call_runs_once_over_the_group():
    """Test that a function declared to run over the whole group gets arrays of all members in one call."""
    BatchElement.shapes.clear()
    group = GroupElement('herd', element=BatchElement, size=1000)
    trajectory = simulate(System(elements=[group]), until=0.03, dt=0.01)
    assert set(BatchElement.shapes) == {(1000,)}
    assert np.asarray(trajectory.values('herd.n'))[-1] == pytest.approx(np.full(1000, np.exp(0.015)), rel=1e-4)


class MeanFieldElement(BaseCountElement):
    """Element whose rate runs once over a whole group and gives every member the mean of all members."""

    def _get_laws(self, s):
        """Return the relaxation towards the mean."""
        return [RateLaw(s.n, CallSpec(lambda t, n: np.full(n.size, n.mean()) - n, (s.n,), vectorized=True))]


def test_vectorized_call_reads_every_member_in_the_jacobian_pattern():
    """Test that a call over the whole group may read any member, so its Jacobian pattern couples all members."""
    group = GroupElement('herd', element=MeanFieldElement, size=4)
    problem = PassCompiler(backend=NumpyBackend()).prepare(System(elements=[group])).run().problem_
    assert problem.jacobian.pattern().toarray().all()


@pytest.mark.parametrize(
    'value',
    [CallSpec(lambda t, values: values['r'] * values['n'], relative=True)],
)
def test_call_options_on_numba_run_in_python(value):
    """Test that a call with an option runs on the Numba backend with the warning of the fallback."""

    class CallElement(BaseCountElement):
        """Element with the given rate."""

        def _get_laws(self, s):
            """Return the growth."""
            return [RateLaw(s.n, value)]

    with pytest.warns(BackendWarning, match='RateLaw laws in Python'):
        problem = PassCompiler(backend=NumbaBackend()).prepare(System(elements=[CallElement('a')])).run().problem_
    assert problem.backends['RateLaw'] == 'python'


class ChangeElement(BaseCountElement):
    """Element whose jump changes the count by a number or a function."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        change: object = 1.0,
        n: Annotated[object, StateSpec()] = 1.0,
        r: Annotated[object, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual, n=n, r=r)
        self.change = change

    def _get_laws(self, s):
        """Return the jump."""
        return [JumpLaw(s.r, {s.n: self.change})]


def _final_counts(change, backend, seeds=40):
    """Return the mean final count of a jump process over several seeds."""
    system = System(elements=[ChangeElement('a', change=change)])
    return np.mean(
        [simulate(system, until=20.0, backend=backend, random_state=seed).values('a.n')[-1] for seed in range(seeds)],
    )


def test_jump_change_function_equals_the_number_on_numpy():
    """Test that a jump whose change is a function gives the run of the number that it returns."""
    system = System(elements=[ChangeElement('a', change=lambda t, values: 2.0)])
    reference = System(elements=[ChangeElement('a', change=2.0)])
    runs = [simulate(item, until=20.0, backend=NumpyBackend(), random_state=3) for item in (system, reference)]
    assert np.array_equal(runs[0].values('a.n'), runs[1].values('a.n'))


def test_jump_change_function_reads_the_state_at_each_firing():
    """Test that each firing applies the change the function returns at the state of that firing."""
    change = CallSpec(lambda t, values: values['n'], relative=True)
    system = System(elements=[ChangeElement('a', change=change)])
    trajectory = simulate(system, until=20.0, backend=NumpyBackend(), random_state=0)
    counts = np.unique(trajectory.values('a.n'))
    assert set(np.log2(counts)) <= set(range(64))
    assert counts.size > 1


def test_jump_change_function_on_numba_runs_in_python():
    """Test that a jump whose change is a function falls back with the warning on the Numba backend."""
    system = System(elements=[ChangeElement('a', change=lambda t, values: 2.0)])
    with pytest.warns(BackendWarning, match='JumpLaw laws in Python'):
        trajectory = simulate(system, until=20.0, backend=NumbaBackend(), random_state=3)
    assert np.all(np.asarray(trajectory.values('a.n')) % 2 == 1)


class ChainElement(Element):
    """Two-state chain whose matrix is given as symbols or a function."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        matrix: object = None,
        x: Annotated[object, StateSpec(domain=ZZ)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.matrix = matrix
        self.x = x

    def _get_laws(self, s):
        """Return the step."""
        return [TransitionLaw(s.x, self.matrix)]


def _chain(matrix, backend=None):
    """Return the states of a long run of a chain."""
    system = System(elements=[ChainElement('a', matrix=matrix)], time=DiscreteTime())
    return np.asarray(
        simulate(system, until=STEPS, backend=backend or NumpyBackend(), random_state=7).values('a.x'),
        dtype=int,
    )


MATRIX = [[0.9, 0.1], [0.4, 0.6]]


def test_transition_function_matches_the_frequencies_of_the_symbolic_matrix():
    """Test that a function of the matrix gives the stationary frequency of the symbolic matrix."""
    symbolic, function = _chain(MATRIX), _chain(lambda t, values: np.array(MATRIX))
    assert symbolic.mean() == pytest.approx(0.2, abs=0.05)
    assert function.mean() == pytest.approx(0.2, abs=0.05)


def test_transition_function_on_numba_runs_in_python():
    """Test that a function of the matrix falls back with the warning on the Numba backend."""
    with pytest.warns(BackendWarning, match='UpdateLaw laws in Python'):
        states = _chain(lambda t, values: np.array(MATRIX), NumbaBackend())
    assert states.mean() == pytest.approx(0.2, abs=0.05)


def test_transition_function_rejects_a_matrix_that_is_not_stochastic():
    """Test that a function that returns rows that do not sum to one raises an error."""
    with pytest.raises(ValidationError, match='rows sum to 1'):
        _chain(lambda t, values: np.array([[0.5, 0.1], [0.4, 0.6]]))


class WaitingElement(BaseCountElement):
    """Element that counts renewals after waiting times given as a draw or a function."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        waiting: object = None,
        n: Annotated[object, StateSpec()] = 0.0,
        r: Annotated[object, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual, n=n, r=r)
        self.waiting = waiting

    def _get_laws(self, s):
        """Return the rate and the renewal."""
        return [RateLaw(s.n, 0.0), JumpLaw(waiting=self.waiting, reset={s.n: s.n + 1}, label='renewal')]


def _waiting_sampler(rng, size, t, n, r):
    """Return constant waiting times, checking the values it receives."""
    assert np.shape(t) == (size,)
    assert np.ndim(n) in {0, 1}
    return rng.uniform(r, r) * np.ones(size)


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_renewal_sampler_counts_like_its_constant_waiting_time(backend):
    """Test that a waiting time given as a function renews like the constant it returns."""
    system = System(elements=[WaitingElement('a', waiting=_waiting_sampler)])
    solver = RK4Solver(dt=0.01)
    with_backend = simulate(system, until=10.2, solver=solver, backend=backend(), random_state=0)
    assert with_backend.values('a.n')[-1] == pytest.approx(20.0)


def test_jump_rejects_a_function_in_the_reset_of_a_renewal():
    """Test that a renewal says that its reset is symbolic."""
    with pytest.raises(ValidationError, match='takes symbolic values'):
        JumpLaw(waiting=_waiting_sampler, reset={SR.var('n'): lambda t, values: 0.0}).prepare()


class SumElement(BaseCountElement):
    """Member whose rate reads the sum of its group."""

    def _get_laws(self, s):
        """Return the rate."""
        return [RateLaw(s.n, total(s.n))]


class ReducedElement(BaseCountElement):
    """Member whose rate reads the sum of its group through a function."""

    def _get_laws(self, s):
        """Return the rate."""
        return [RateLaw(s.n, reduction(np.sum, s.n))]


def _reduced(element, backend):
    """Return the final values of a group that grows by a reduction over it."""
    group = GroupElement('herd', element=element, size=3, values={'n': [1.0, 2.0, 4.0]})
    trajectory = simulate(System(elements=[group]), until=0.5, dt=0.01, backend=backend)
    return np.asarray(trajectory.values('herd.n'))[-1]


def test_reduction_function_equals_the_symbolic_reduction():
    """Test that a reduction by a function gives the trajectory of the symbolic reduction."""
    assert _reduced(ReducedElement, NumpyBackend()) == pytest.approx(_reduced(SumElement, NumpyBackend()), rel=1e-9)


def test_reduction_function_on_numba_runs_in_python():
    """Test that a reduction by a function falls back with the warning on the Numba backend."""
    with pytest.warns(BackendWarning, match='the function reduction'):
        values = _reduced(ReducedElement, NumbaBackend())
    assert values == pytest.approx(_reduced(SumElement, NumpyBackend()), rel=1e-9)


def _system_output(t, values):
    """Return the value of a system trained elsewhere from the values of a system."""
    return 2.0 * values['a.n'] + t


def test_observable_function_records_its_value_at_every_frame():
    """Test that an observable given as a function gives the expression that it computes at every frame."""
    observables = [ObservableSpec('learned', function=_system_output)]
    for backend in (NumpyBackend(), NumbaBackend()):
        system = System(elements=[SymbolicElement('a')])
        trajectory = simulate(system, until=1.0, dt=0.1, observables=observables, backend=backend)
        expected = 2.0 * np.asarray(trajectory.values('a.n')) + np.asarray(trajectory.times)
        assert np.asarray(trajectory.values('learned')) == pytest.approx(expected)


def _burgers_speed(t, values):
    """Return the wave speed of the Burgers flux."""
    return values['u']


class BurgersElement(Element):
    """Cell of a Burgers field, with its wave speed from the Jacobian."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u

    def _get_laws(self, s):
        """Return the flux."""
        return [FluxLaw(s.u, s.u**2 / 2)]


class SpeedElement(BurgersElement):
    """Cell of a Burgers field, with its wave speed from a function."""

    def _get_laws(self, s):
        """Return the flux."""
        return [FluxLaw(s.u, s.u**2 / 2, speed=_burgers_speed)]


def _burgers(element):
    """Return the final values of a Burgers field."""
    n = 40
    centers = (np.arange(n) + 0.5) * 2 * np.pi / n
    field = FieldElement(
        'line',
        element=element,
        space=LatticeSpace(n, spacing=2 * np.pi / n),
        values={'u': 1.0 + 0.5 * np.sin(centers)},
        discretization='finite-volume',
    )
    return np.asarray(simulate(System(elements=[field]), until=0.5, dt=0.5, solver=SSPRKSolver()).values('line.u'))[-1]


def test_flux_law_speed_function_equals_the_eigenvalues():
    """Test that a wave speed given as a function gives the run of the same speed from the Jacobian."""
    assert _burgers(SpeedElement) == pytest.approx(_burgers(BurgersElement), abs=1e-9)


def test_flux_law_rejects_a_function_as_a_flux_or_a_speed_of_godunov():
    """Test that a flux says it is symbolic only and names the callable alternative."""
    u = SR.var('u')
    with pytest.raises(ValidationError, match=r'symbolic only.*speed='):
        FluxLaw(u, lambda t, values: values['u'], riemann='godunov').prepare()
    with pytest.raises(ValidationError, match='takes no `speed`'):
        FluxLaw(u, u**2 / 2, riemann='godunov', speed=lambda t, values: values['u']).prepare()
