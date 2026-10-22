"""Test the Jacobian structure that ScipyODE passes to its implicit methods, at every start of the integrator."""

import warnings

import numpy as np
import pytest

from skcomplex.core.base import SolverError, ValidationError
from skcomplex.core.compile import compile_system
from skcomplex.core.model import (
    Element,
    Event,
    Field,
    Interaction,
    LatticeSpace,
    Rate,
    System,
    delayed,
    laplacian,
    total,
)
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4, ScipyODE
from skcomplex.models.biology.ecology import FisherKPP, fisher_kpp

SIZE = 400

Threshold = Interaction.define(
    'Threshold',
    roles=['line'],
    parameters={'level': 0.0},
    laws=lambda s: [Event(total(s.line.u) - s.level, terminal=True)],
)

DiffusiveHutchinson = Element.define(
    'DiffusiveHutchinson',
    states={'u': 0.0},
    parameters={'D': 1.0, 'r': 1.0},
    laws=lambda s: [Rate(s.u, s.D * laplacian(s.u) + s.r * s.u * (1 - delayed(s.u, 1.0)))],
)


def _stiff_fisher(size=SIZE):
    """Return Fisher-KPP whose diffusion makes the largest eigenvalue about 1600."""
    return compile_system(fisher_kpp(size, D=1.0, spacing=0.05))


def _advance(problem, until=1.0, p=None, **options):
    """Return the state of a problem after one call of ScipyODE with options, and the stepper."""
    stepper = ScipyODE(**options)
    stepper.initialize(problem, np.random.default_rng(0))
    parameters = problem.initial_parameters if p is None else p
    return stepper.advance(0.0, problem.initial_state, parameters, until), stepper


def _relative(first, second):
    """Return the relative difference of two states."""
    return np.linalg.norm(first - second) / np.linalg.norm(second)


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
        components=[
            Field(
                'line',
                FisherKPP,
                LatticeSpace(50, spacing=0.25),
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
    Member = Element.define('Mixed', states={'x': 0.0}, laws=lambda s: [Rate(s.x, -s.x + 0.1 * total(s.x))])
    problem = compile_system(System(components=[Member.group('group', 10, x=np.arange(10.0))]))
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
        ScipyODE(jacobian='sparse').initialize(problem, np.random.default_rng(0))
    with pytest.raises(SolverError, match='BDF'):
        ScipyODE(method='Euler').initialize(problem, np.random.default_rng(0))


def _callable_rate(t, values):
    """Return the decay of the line."""
    return -values['line.u']


def test_exact_option_with_callable_law_raises_on_initialize():
    """Test that asking for the exact Jacobian of a callable law raises when initializing."""
    Decay = Element.define('Decay', states={'u': 0.0}, laws=lambda s: [Rate(s.u, _callable_rate)])
    problem = compile_system(System(components=[Field('line', Decay, LatticeSpace(10))]))
    with pytest.raises(ValidationError, match="jacobian='auto'"):
        ScipyODE(method='BDF', jacobian='exact').initialize(problem, np.random.default_rng(0))


def test_exact_jacobian_follows_parameter_changes():
    """Test that a parameter change between calls reaches the exact Jacobian of the next call."""
    problem = _stiff_fisher(100)
    p = problem.initial_parameters.copy()
    states = {}
    for option in ['exact', 'dense']:
        stepper = ScipyODE(method='BDF', rtol=1e-8, jacobian=option)
        stepper.initialize(problem, np.random.default_rng(0))
        x = stepper.advance(0.0, problem.initial_state, p, 0.5)
        changed = p.copy()
        changed[problem.parameters.entry('line.D').indices] = 3.0
        states[option] = stepper.advance(0.5, x, changed, 1.0)
        if option == 'exact':
            jacobian = stepper._solver.jac(0.5, x).toarray()
            np.testing.assert_allclose(jacobian, problem.jacobian(0.5, x, changed).toarray())
    assert _relative(states['exact'], states['dense']) < 1e-5


def _event_time(size, jacobian):
    """Return the time of the terminal event of Fisher-KPP when its total grows by half."""
    initial = (np.arange(size) < 0.05 * size).astype(float)
    space = LatticeSpace(size, spacing=0.05)
    field = Field('line', FisherKPP, space, boundary={'u': 'zero-flux'}, values={'u': initial})
    simulation = Simulation(
        System(components=[field], interactions=[Threshold('threshold', line='line', level=1.5 * initial.sum())]),
        solver=ScipyODE(method='BDF', jacobian=jacobian),
        dt=1.0,
    )
    return simulation.run(until=10).trajectory_.times[-1], simulation


def test_event_on_stiff_field_keeps_the_pattern():
    """Test that BDF with the pattern finds the event of a stiff field at the time of the dense path."""
    time, simulation = _event_time(SIZE, 'auto')
    dense, _ = _event_time(SIZE, 'dense')
    assert 0 < time < 10
    assert abs(time - dense) < 1e-6
    assert 'jac_sparsity' in simulation.stepper_.jacobian_options_


def _hutchinson(size, jacobian):
    """Return the final state of a diffusive Hutchinson equation run with BDF."""
    initial = 0.5 + 0.1 * np.cos(np.linspace(0, 2 * np.pi, size))
    field = Field(
        'line',
        DiffusiveHutchinson,
        LatticeSpace(size, spacing=0.05),
        boundary={'u': 'zero-flux'},
        values={'u': initial, 'r': 1.2},
    )
    simulation = Simulation(System(components=[field]), solver=ScipyODE(method='BDF', jacobian=jacobian), dt=1.0)
    return simulation.run(until=4).trajectory_.states[-1]


def test_delayed_stiff_field_keeps_the_pattern():
    """Test that BDF with the pattern integrates a delayed stiff field like the dense path."""
    assert _relative(_hutchinson(100, 'auto'), _hutchinson(100, 'dense')) < 1e-5


def test_stability_warning_names_the_stiff_path():
    """Test that the stability warning of a fixed-step explicit stepper names BDF and Radau."""
    problem = _stiff_fisher(50)
    with pytest.warns(UserWarning, match=r"method='BDF'.*method='Radau'"):
        RK4(dt=0.01).initialize(problem, np.random.default_rng(0))


def test_adaptive_explicit_method_warns_once_on_stiff_field():
    """Test that RK45 warns once on the first frame that spans more than 1000 stable steps, and not later."""
    problem = _stiff_fisher(100)
    assert problem.stable_step == pytest.approx(1.25e-3)
    stepper = ScipyODE(method='RK45')
    stepper.initialize(problem, np.random.default_rng(0))
    with pytest.warns(UserWarning, match=r"method='BDF'.*method='Radau'"):
        x = stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 2.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        stepper.advance(2.0, x, problem.initial_parameters, 4.0)
