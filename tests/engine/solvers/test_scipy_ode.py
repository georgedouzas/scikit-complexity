"""Test the `ScipyODESolver` class."""

import dataclasses
import time
import warnings
from dataclasses import replace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import ceil, sin
from scipy.integrate import solve_ivp

from skcomplex.core.elements import Element, FieldElement, GroupElement
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import CrossingLaw, RateLaw
from skcomplex.core.laws.utils import delayed, laplacian, mean, step, total, where
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.backends import NumpyBackend
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import CrankNicolsonSolver, RK4Solver, ScipyODESolver
from skcomplex.errors import SolverError, ValidationError
from skcomplex.models import create, run
from skcomplex.models.biology.ecology import (
    ExponentialGrowthInteraction,
    FisherKPPElement,
    PopulationElement,
    fisher_kpp,
)
from skcomplex.models.physics.heat import heat_diffusion
from skcomplex.models.physics.mechanics import harmonic_oscillator
from skcomplex.specs import ParameterSpec, RoleSpec, RunStateSpec, StateSpec, StepSpec, VisualSpec


def _growth_problem():
    """Return the compiled problem of exponential growth."""
    system = System(
        elements=[PopulationElement('a', n=1)],
        interactions=[ExponentialGrowthInteraction('g', target='a', r=1.0)],
    )
    return compile_system(system)


@pytest.mark.parametrize('method', ['RK45', 'DOP853', 'LSODA'])
def test_scipy_ode_matches_exponential(method):
    """Test that the ODE stepper integrates exponential growth accurately."""
    problem = _growth_problem()
    stepper = ScipyODESolver(method=method, rtol=1e-9, atol=1e-12)
    stepper.set_params(random_state=np.random.default_rng(0)).check().prepare(problem)
    state = stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 2.0))
    assert state[0] == pytest.approx(np.exp(2.0), rel=1e-6)


class ExplosionInteraction(Interaction):
    """Growth `dn/dt = n**2`, which blows up in finite time."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return the rate `n**2`."""
        return [RateLaw(s.target.n, s.target.n**2)]


def test_scipy_ode_failure_raises_error():
    """Test that an integrator failure raises a named error."""
    problem = compile_system(
        System(elements=[PopulationElement('a', n=1)], interactions=[ExplosionInteraction('boom', target='a')]),
    )
    stepper = ScipyODESolver()
    stepper.set_params(random_state=np.random.default_rng(0)).check().prepare(problem)
    with pytest.raises(SolverError, match='failed'):
        stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 2.0))


def test_simulation_runs_stiff_model_with_default_solver():
    """Test that a stiff model built directly runs fast with the default solver and matches its registered solver."""
    start = time.perf_counter()
    default = Simulation(create('nanoparticle_corona')).check().run(until=10800)
    assert time.perf_counter() - start < 5
    registered = run('nanoparticle_corona', settings={'until': 10800})
    np.testing.assert_allclose(default.state_, registered.state_, rtol=1e-4, atol=1e-12)


def test_simulation_default_solver_keeps_accuracy_of_non_stiff_model():
    """Test that a harmonic oscillator run with the default solver matches the analytic solution within 1e-5."""
    simulation = Simulation(harmonic_oscillator()).check().run(until=10)
    np.testing.assert_allclose(simulation.state_, [np.cos(10), -np.sin(10)], rtol=1e-5)


def _relaxing_group(size):
    """Return a group whose members relax to their mean, so every member reads every other one."""

    class RelaxElement(Element):
        """Relax element."""

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
            """Return the laws of the relax element."""
            return [RateLaw(s.x, -(s.x - mean(s.x)))]

    return System(
        elements=[GroupElement('cells', element=RelaxElement, size=size, values={'x': np.linspace(0.0, 1.0, size)})],
    )


def test_default_solver_leaves_lsoda_for_states_its_dense_matrix_cannot_hold():
    """Test that 10^4 coupled states leave LSODA, a banded field of 6400 states keeps it, and LSODA names the limit."""
    simulation = Simulation(_relaxing_group(10_000), dt=0.5).check().run(until=1)
    assert simulation.stepper_.method_ != 'LSODA'
    start = np.linspace(0.0, 1.0, 10_000)
    np.testing.assert_allclose(simulation.state_, 0.5 + (start - 0.5) * np.exp(-1), atol=1e-5)
    assert Simulation(_relaxing_group(10), dt=0.5).check().run(until=1).stepper_.method_ == 'LSODA'
    banded = Simulation(heat_diffusion(shape=(80, 80)), dt=0.1).check().run(until=0.2)
    assert banded.stepper_.method_ == 'LSODA'
    assert 'lband' in banded.stepper_.jacobian_options_
    with pytest.raises(SolverError, match='dense 10000 x 10000 matrix'):
        Simulation(_relaxing_group(10_000), dt=0.5, solver=ScipyODESolver(method='LSODA')).check().run(until=1)


class TubElement(Element):
    """Tub element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        water: Annotated[object, StateSpec()] = 20.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.water = water

    def _get_laws(self, s):
        """Return the laws of the tub element."""
        return [RateLaw(s.water, 5 * step(s.t, 5) - where(s.water > 0, 5, 0))]


def test_default_solver_leaves_lsoda_when_a_state_rests_on_a_discontinuity():
    """Test that a group of tubs resting empty on their drain switch finishes with RK45, and explicit LSODA stalls."""
    tubs = System(elements=[GroupElement('tubs', element=TubElement, size=1, values={'water': [20.0]})])
    with pytest.warns(UserWarning, match='LSODA stalled'):
        simulation = Simulation(tubs, dt=1).check().run(until=6)
    assert simulation.stepper_.method_ == 'RK45'
    np.testing.assert_allclose(np.ravel(simulation.trajectory_.values('tubs.water'))[:5], [20, 15, 10, 5, 0], atol=1e-6)
    with pytest.raises(SolverError, match='stalled'):
        Simulation(tubs, dt=1, solver=ScipyODESolver(method='LSODA')).check().run(until=6)


def test_default_solver_starts_from_rest_with_a_vanishing_drift():
    """Test that an oscillator at rest, forced by `sin(t)`, leaves its zero drift at time 0 and follows the solution."""

    class ForcedElement(Element):
        """Forced element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 0.0,
            v: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x
            self.v = v

        def _get_laws(self, s):
            """Return the laws of the forced element."""
            return [RateLaw(s.x, s.v), RateLaw(s.v, -4 * s.x + sin(s.t))]

    simulation = Simulation(System(elements=[ForcedElement('oscillator')]), dt=0.01).check().run(until=5)
    times = simulation.trajectory_.times
    expected = (np.sin(times) - np.sin(2 * times) / 2) / 3
    np.testing.assert_allclose(simulation.trajectory_.values('oscillator.x'), expected, atol=1e-6)


def _decay():
    """Return the system of exponential decay."""
    return System(
        elements=[PopulationElement('a', n=1)],
        interactions=[ExponentialGrowthInteraction('g', target='a', r=-1.0)],
    )


def test_scipy_ode_keeps_integrator_across_frames():
    """Test that 1000 frames of ScipyODE evaluate the drift at most twice as often as one SciPy integration."""
    problem = compile_system(_decay())
    calls = []

    def counted(t, x, p):
        calls.append(t)
        return problem.drift(t, x, p)

    trajectory = Simulation(dataclasses.replace(problem, drift=counted), dt=0.01).check().run(until=10).trajectory_
    frames = len(calls)
    calls.clear()
    p = problem.initial_parameters
    solve_ivp(lambda t, x: counted(t, x, p), (0.0, 10.0), problem.initial_state, rtol=1e-6, atol=1e-9)
    assert len(trajectory.times) == 1001
    assert frames <= 2 * len(calls)
    assert trajectory.values('a.n')[-1] == pytest.approx(np.exp(-10.0), rel=1e-4)


def test_scipy_ode_restarts_on_parameter_change():
    """Test that a decay rate set to zero at time 5 keeps the state constant from then on."""
    simulation = Simulation(_decay(), dt=0.01).check().run(until=5)
    simulation.update_parameters({'g.r': 0.0})
    values = simulation.extend(until=10).trajectory_.values('a.n')
    assert values[500] == pytest.approx(np.exp(-5.0), rel=1e-5)
    assert np.all(values[500:] == values[500])


class ChatterElement(Element):
    """Chatter element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        w: Annotated[object, StateSpec()] = 20.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.w = w

    def _get_laws(self, s):
        """Return the laws of the chatter element."""
        return [RateLaw(s.w, 5 - 10 * ceil(s.w / 100))]


def test_default_solver_names_a_state_sliding_along_a_discontinuity():
    """Test that a state sliding along a switch no event sees ends in a named error instead of a run without end."""
    with pytest.warns(UserWarning, match='LSODA stalled'), pytest.raises(SolverError, match='slides along'):
        Simulation(System(elements=[ChatterElement('tub')]), dt=1).check().run(until=6)


def test_default_solver_names_a_blow_up():
    """Test that a solution that blows up in finite time ends in an error that names the blow-up."""
    with pytest.raises(SolverError, match='blows up'):
        Simulation(create('takens_bogdanov_form', x=5.0, y=5.0, beta1=0.1, beta2=-1.0), dt=0.1).check().run(until=50)


def test_default_solver_steps_large_diffusive_fields_explicitly():
    """Test that a large field whose frames span a few stable steps runs with RK45, not a costly implicit method."""
    simulation = Simulation(create('fitzhugh_nagumo'), dt=2.0).check().run(until=4)
    assert simulation.stepper_.method_ == 'RK45'


STIFF_SIZE = 4000


def _stiff_fisher_problem():
    """Return Fisher-KPP on 4000 points whose diffusion makes the largest eigenvalue about 1600."""
    return compile_system(fisher_kpp(STIFF_SIZE, D=1.0, spacing=0.05), backend=NumpyBackend())


def _counted(problem):
    """Return a problem whose drift counts its evaluations, with the counter."""
    calls = [0]
    drift = problem.drift

    def counting(t, x, p):
        calls[0] += 1
        return drift(t, x, p)

    return replace(problem, drift=counting), calls


def _timed_scipy(problem, method, jacobian, until=10.0):
    """Return the duration, the final state, the drift evaluations, and the stepper of one call of ScipyODE."""
    problem, calls = _counted(problem)
    stepper = ScipyODESolver(method=method, rtol=1e-6, jacobian=jacobian)
    stepper.set_params(random_state=np.random.default_rng(0)).check().prepare(problem)
    start = time.perf_counter()
    state = stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, until))
    return time.perf_counter() - start, state, calls[0], stepper


class StiffThresholdInteraction(Interaction):
    """Stiff threshold interaction."""

    def __init__(
        self,
        label: str,
        *,
        line: Annotated[object, RoleSpec()],
        level: Annotated[object, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.line = line
        self.level = level

    def _get_laws(self, s):
        """Return the laws of the stiff threshold interaction."""
        return [CrossingLaw(total(s.line.u) - s.level, terminal=True)]


def _event_run(jacobian):
    """Return the duration and the event time of stiff Fisher-KPP that stops when its total grows by half."""
    line = fisher_kpp(STIFF_SIZE, D=1.0, spacing=0.05).elements[0]
    level = 1.5 * line.values['u'].sum()
    system = System(elements=[line], interactions=[StiffThresholdInteraction('threshold', line='line', level=level)])
    simulation = Simulation(
        system,
        solver=ScipyODESolver(method='BDF', jacobian=jacobian),
        dt=1.0,
        backend=NumpyBackend(),
    ).check()
    start = time.perf_counter()
    times = simulation.run(until=10.0).trajectory_.times
    return time.perf_counter() - start, times[-1]


class DiffusiveHutchinsonElement(Element):
    """Diffusive hutchinson element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[object, StateSpec()] = 0.0,
        D: Annotated[object, ParameterSpec()] = 1.0,
        r: Annotated[object, ParameterSpec()] = 1.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.D = D
        self.r = r

    def _get_laws(self, s):
        """Return the laws of the diffusive hutchinson element."""
        return [RateLaw(s.u, s.D * laplacian(s.u) + s.r * s.u * (1 - delayed(s.u, 1.0)))]


def _hutchinson_run(jacobian):
    """Return the duration and the final state of a diffusive Hutchinson equation on 2000 points with BDF."""
    size = 2000
    initial = 0.5 + 0.1 * np.cos(np.linspace(0, 2 * np.pi, size))
    field = FieldElement(
        'line',
        element=DiffusiveHutchinsonElement,
        space=LatticeSpace(size, spacing=0.05),
        values={'u': initial},
    )
    simulation = Simulation(
        System(elements=[field]),
        solver=ScipyODESolver(method='BDF', jacobian=jacobian),
        dt=1.0,
        backend=NumpyBackend(),
    ).check()
    start = time.perf_counter()
    state = simulation.run(until=5.0).trajectory_.states[-1]
    return time.perf_counter() - start, state


SIZE = 400


class ThresholdInteraction(Interaction):
    """Threshold interaction."""

    def __init__(
        self,
        label: str,
        *,
        line: Annotated[object, RoleSpec()],
        level: Annotated[object, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.line = line
        self.level = level

    def _get_laws(self, s):
        """Return the laws of the threshold interaction."""
        return [CrossingLaw(total(s.line.u) - s.level, terminal=True)]


class UnitRateHutchinsonElement(Element):
    """Unit rate hutchinson element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[object, StateSpec()] = 0.0,
        D: Annotated[object, ParameterSpec()] = 1.0,
        r: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.D = D
        self.r = r

    def _get_laws(self, s):
        """Return the laws of the unit rate hutchinson element."""
        return [RateLaw(s.u, s.D * laplacian(s.u) + s.r * s.u * (1 - delayed(s.u, 1.0)))]


def _stiff_fisher(size=SIZE):
    """Return Fisher-KPP whose diffusion makes the largest eigenvalue about 1600."""
    return compile_system(fisher_kpp(size, D=1.0, spacing=0.05))


def _advance(problem, until=1.0, p=None, **options):
    """Return the state of a problem after one call of ScipyODE with options, and the stepper."""
    stepper = ScipyODESolver(**options)
    stepper.set_params(random_state=np.random.default_rng(0)).check().prepare(problem)
    parameters = problem.initial_parameters if p is None else p
    return stepper.run(StepSpec(0.0, problem.initial_state, parameters, until)), stepper


def _relative(first, second):
    """Return the relative difference of two states."""
    return np.linalg.norm(first - second) / np.linalg.norm(second)


def _callable_rate(t, values):
    """Return the decay of the line."""
    return -values['line.u']


def _event_time(size, jacobian):
    """Return the time of the terminal event of Fisher-KPP when its total grows by half."""
    initial = (np.arange(size) < 0.05 * size).astype(float)
    space = LatticeSpace(size, spacing=0.05)
    field = FieldElement(
        'line',
        element=FisherKPPElement,
        space=space,
        boundary={'u': 'zero-flux'},
        values={'u': initial},
    )
    simulation = Simulation(
        System(
            elements=[field],
            interactions=[ThresholdInteraction('threshold', line='line', level=1.5 * initial.sum())],
        ),
        solver=ScipyODESolver(method='BDF', jacobian=jacobian),
        dt=1.0,
    ).check()
    return simulation.run(until=10).trajectory_.times[-1], simulation


def _hutchinson(size, jacobian):
    """Return the final state of a diffusive Hutchinson equation run with BDF."""
    initial = 0.5 + 0.1 * np.cos(np.linspace(0, 2 * np.pi, size))
    field = FieldElement(
        'line',
        element=UnitRateHutchinsonElement,
        space=LatticeSpace(size, spacing=0.05),
        boundary={'u': 'zero-flux'},
        values={'u': initial, 'r': 1.2},
    )
    simulation = Simulation(
        System(elements=[field]),
        solver=ScipyODESolver(method='BDF', jacobian=jacobian),
        dt=1.0,
    ).check()
    return simulation.run(until=4).trajectory_.states[-1]


@pytest.mark.benchmark
def test_sparse_jacobians_speed_up_stiff_fields(record_property):
    """Test that the pattern, Radau, and the exact Jacobian are far faster than BDF on the dense path."""
    problem = _stiff_fisher_problem()
    _timed_scipy(problem, 'BDF', 'auto', until=0.1)
    pattern, reference, pattern_calls, stepper = _timed_scipy(problem, 'BDF', 'auto')
    dense, dense_state, _, _ = _timed_scipy(problem, 'BDF', 'dense')
    radau, radau_state, _, _ = _timed_scipy(problem, 'Radau', 'auto')
    exact, exact_state, exact_calls, exact_stepper = _timed_scipy(problem, 'BDF', 'exact')
    split = Simulation(problem, solver=CrankNicolsonSolver(dt=0.01), dt=10.0, record=False).check()
    start = time.perf_counter()
    split.run(until=10.0)
    crank_nicolson = time.perf_counter() - start
    for name, value in [('pattern', pattern), ('radau', radau), ('exact', exact), ('crank_nicolson', crank_nicolson)]:
        record_property(f'dense_over_{name}', dense / value)
    assert dense >= 20 * pattern
    assert radau * 10 <= dense
    assert exact * 10 <= dense
    for solved in [dense_state, radau_state, exact_state]:
        assert np.linalg.norm(solved - reference) / np.linalg.norm(reference) < 1e-5
    assert pattern_calls > stepper._solver.nfev
    assert exact_calls == exact_stepper._solver.nfev


@pytest.mark.benchmark
def test_banded_lsoda_speeds_up_stiff_fields(record_property):
    """Test that LSODA with half bandwidths of 1 is at least 10 times faster than without them."""
    problem = _stiff_fisher_problem()
    banded, state, _, stepper = _timed_scipy(problem, 'LSODA', 'auto')
    dense, dense_state, _, _ = _timed_scipy(problem, 'LSODA', 'dense')
    record_property('dense_over_banded', dense / banded)
    assert stepper.jacobian_options_ == {'lband': 1, 'uband': 1}
    assert banded * 10 <= dense
    assert np.linalg.norm(state - dense_state) / np.linalg.norm(dense_state) < 1e-5


@pytest.mark.benchmark
def test_sparse_jacobians_speed_up_events_on_stiff_fields(record_property):
    """Test that the pattern locates the event of a stiff field at the dense path's time, at least 10 times faster."""
    _event_run('auto')
    pattern, time_pattern = _event_run('auto')
    dense, time_dense = _event_run('dense')
    record_property('dense_over_pattern', dense / pattern)
    assert 0 < time_pattern < 10
    assert abs(time_pattern - time_dense) < 1e-6
    assert pattern * 10 <= dense


@pytest.mark.benchmark
def test_sparse_jacobians_keep_delayed_stiff_fields_accurate(record_property):
    """Test that a delayed stiff field with the pattern agrees with the dense path."""
    pattern, state = _hutchinson_run('auto')
    dense, dense_state = _hutchinson_run('dense')
    record_property('dense_over_pattern', dense / pattern)
    assert np.linalg.norm(state - dense_state) / np.linalg.norm(dense_state) < 1e-5


def test_bdf_receives_the_pattern_and_agrees_with_the_dense_path():
    """Test that BDF receives the pattern by default and agrees with the dense path."""
    problem = _stiff_fisher()
    sparse, stepper = _advance(problem, method='BDF')
    assert stepper.jacobian_options_['jac_sparsity'].nnz == 3 * SIZE - 2
    dense, dense_stepper = _advance(problem, method='BDF', jacobian='dense')
    assert dense_stepper.jacobian_options_ == {}
    assert dense_stepper.jacobian_pattern_ is None
    assert _relative(sparse, dense) < 1e-5


def test_radau_and_exact_agree_with_the_pattern():
    """Test that Radau with the pattern and BDF with the exact Jacobian agree with BDF and the pattern."""
    problem = _stiff_fisher()
    reference, _ = _advance(problem, method='BDF', rtol=1e-8)
    radau, _ = _advance(problem, method='Radau', rtol=1e-8)
    exact, stepper = _advance(problem, method='BDF', rtol=1e-8, jacobian='exact')
    assert stepper.jacobian_options_ == {}
    assert stepper._solver.njev >= 1
    assert _relative(radau, reference) < 1e-5
    assert _relative(exact, reference) < 1e-5


def test_lsoda_receives_half_bandwidths():
    """Test that LSODA receives half bandwidths of 1 for a three-point Laplacian with zero-flux edges."""
    problem = _stiff_fisher()
    banded, stepper = _advance(problem, method='LSODA')
    assert stepper.jacobian_options_ == {'lband': 1, 'uband': 1}
    exact, _ = _advance(problem, method='LSODA', jacobian='exact')
    reference, _ = _advance(problem, method='BDF', rtol=1e-8)
    assert _relative(banded, reference) < 1e-5
    assert _relative(exact, reference) < 1e-5


def test_lsoda_without_narrow_band_receives_nothing():
    """Test that LSODA receives no bandwidths when periodic edges make the band as wide as the line."""
    system = System(
        elements=[
            FieldElement(
                'line',
                element=FisherKPPElement,
                space=LatticeSpace(50, spacing=0.25),
                boundary={'u': 'periodic'},
                values={'u': np.linspace(0, 1, 50)},
            ),
        ],
    )
    _, stepper = _advance(compile_system(system), method='LSODA')
    assert stepper.jacobian_options_ == {}
    assert stepper.jacobian_pattern_ is not None


def test_dense_pattern_is_not_passed():
    """Test that a pattern denser than one half is not passed."""

    class MixedElement(Element):
        """Mixed element."""

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
            """Return the laws of the mixed element."""
            return [RateLaw(s.x, -s.x + 0.1 * total(s.x))]

    MemberElement = MixedElement
    problem = compile_system(
        System(elements=[GroupElement('group', element=MemberElement, size=10, values={'x': np.arange(10.0)})]),
    )
    _, stepper = _advance(problem, method='BDF')
    assert stepper.jacobian_options_ == {}
    assert stepper.jacobian_pattern_ is None


@pytest.mark.parametrize('jacobian', ['auto', 'exact', 'dense'])
def test_explicit_methods_ignore_the_option(jacobian):
    """Test that RK45 builds no structure and returns the same trajectory with every option."""
    problem = compile_system(fisher_kpp(100))
    state, stepper = _advance(problem, method='RK45', jacobian=jacobian)
    reference, _ = _advance(problem, method='RK45')
    assert stepper.jacobian_pattern_ is None
    assert stepper.jacobian_options_ == {}
    np.testing.assert_array_equal(state, reference)


def test_unknown_option_raises():
    """Test that an unknown Jacobian option or method raises an error that lists the options."""
    problem = compile_system(fisher_kpp(10))
    with pytest.raises(SolverError, match="'auto', 'exact', 'dense'"):
        ScipyODESolver(jacobian='sparse', random_state=0).check().prepare(problem)
    with pytest.raises(SolverError, match='BDF'):
        ScipyODESolver(method='Euler', random_state=0).check().prepare(problem)


def test_exact_option_with_callable_law_raises_on_initialize():
    """Test that asking for the exact Jacobian of a callable law raises when initializing."""

    class DecayElement(Element):
        """Decay element."""

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
            """Return the laws of the decay element."""
            return [RateLaw(s.u, _callable_rate)]

    problem = compile_system(System(elements=[FieldElement('line', element=DecayElement, space=LatticeSpace(10))]))
    with pytest.raises(ValidationError, match="jacobian='auto'"):
        ScipyODESolver(method='BDF', jacobian='exact', random_state=0).check().prepare(problem)


def test_exact_jacobian_follows_parameter_changes():
    """Test that a parameter change between calls reaches the exact Jacobian of the next call."""
    problem = _stiff_fisher(100)
    p = problem.initial_parameters.copy()
    states = {}
    for option in ['exact', 'dense']:
        stepper = ScipyODESolver(method='BDF', rtol=1e-8, jacobian=option)
        stepper.set_params(random_state=np.random.default_rng(0)).check().prepare(problem)
        x = stepper.run(StepSpec(0.0, problem.initial_state, p, 0.5))
        changed = p.copy()
        changed[problem.parameters.entry('line.D').indices] = 3.0
        states[option] = stepper.run(StepSpec(0.5, x, changed, 1.0))
        if option == 'exact':
            jacobian = stepper._solver.jac(0.5, x).toarray()
            exact = problem.jacobian.prepare(problem).run(RunStateSpec(0.5, x, changed)).toarray()
            np.testing.assert_allclose(jacobian, exact)
    assert _relative(states['exact'], states['dense']) < 1e-5


def test_event_on_stiff_field_keeps_the_pattern():
    """Test that BDF with the pattern finds the event of a stiff field at the time of the dense path."""
    time, simulation = _event_time(SIZE, 'auto')
    dense, _ = _event_time(SIZE, 'dense')
    assert 0 < time < 10
    assert abs(time - dense) < 1e-6
    assert 'jac_sparsity' in simulation.stepper_.jacobian_options_


def test_delayed_stiff_field_keeps_the_pattern():
    """Test that BDF with the pattern integrates a delayed stiff field like the dense path."""
    assert _relative(_hutchinson(100, 'auto'), _hutchinson(100, 'dense')) < 1e-5


def test_stability_warning_names_the_stiff_path():
    """Test that the stability warning of a fixed-step explicit stepper names BDF and Radau."""
    problem = _stiff_fisher(50)
    with pytest.warns(UserWarning, match=r"method='BDF'.*method='Radau'"):
        RK4Solver(dt=0.01, random_state=0).check().prepare(problem)


def test_adaptive_explicit_method_warns_once_on_stiff_field():
    """Test that RK45 warns once on the first frame that spans more than 1000 stable steps, and not later."""
    problem = _stiff_fisher(100)
    assert problem.stable_step == pytest.approx(1.25e-3)
    stepper = ScipyODESolver(method='RK45')
    stepper.set_params(random_state=np.random.default_rng(0)).check().prepare(problem)
    with pytest.warns(UserWarning, match=r"method='BDF'.*method='Radau'"):
        x = stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 2.0))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        stepper.run(StepSpec(2.0, x, problem.initial_parameters, 4.0))
