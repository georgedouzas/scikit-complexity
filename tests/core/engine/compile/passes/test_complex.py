"""Test the `_complex` module of `skcomplex.core.engine.compile.passes`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import I, arg, conjugate, exp, gamma, imag_part, real_part

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import CallSpec, ConstraintLaw, EventLaw, JumpLaw, RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import laplacian, mod, total, where
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.solvers import (
    CrankNicolsonSolver,
    ETDRK4Solver,
    IMEX2Solver,
    IMEXEulerSolver,
    RK4Solver,
    ScipyODESolver,
)
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.physics.pattern_formation import GinzburgLandauElement, GinzburgLandauInteraction

from ..conftest import QuadraticElement, phase_system, quadratic_system

STIFF = (IMEXEulerSolver(dt=0.05), IMEX2Solver(dt=0.05), CrankNicolsonSolver(dt=0.05))


def _system(element, laws, **settings):
    """Return a system of one element whose laws are a function of its namespace and the time."""
    rule = Interaction('rule', roles={'target': element.label}, laws=lambda s: laws(s.target, s.t))
    return System(elements=[element], interactions=[rule], **settings)


def _readout(**settings):
    """Return the rotation of a complex value with its modulus read into a real value."""
    element = Element('a', values={'z': 2.0 + 0j, 'r': 0.0})
    return _system(element, lambda a, t: [UpdateLaw(a.z, a.z * exp(I * 0.3)), UpdateLaw(a.r, abs(a.z))], **settings)


def _stuart(**settings):
    """Return the Stuart-Landau oscillator `dz/dt = (1 + i) z - |z|^2 z`, whose law is not holomorphic."""
    element = Element('o', values={'z': 0.5 + 0.2j})
    return _system(element, lambda o, t: [RateLaw(o.z, (1 + I) * o.z - abs(o.z) ** 2 * o.z)], **settings)


def _split_stuart(**settings):
    """Return the Stuart-Landau oscillator written with its real and imaginary parts."""

    def laws(o, t):
        square = o.u**2 + o.v**2
        return [RateLaw(o.u, o.u - o.v - square * o.u), RateLaw(o.v, o.u + o.v - square * o.v)]

    return _system(Element('o', values={'u': 0.5, 'v': 0.2}), laws, **settings)


def _ginzburg_landau(shape=(32, 32), discretization='spectral'):
    """Return the complex and split Ginzburg-Landau systems from one small random amplitude."""
    rng = np.random.default_rng(0)
    u, v = 0.01 * rng.uniform(-1, 1, shape), 0.01 * rng.uniform(-1, 1, shape)
    lattice = LatticeSpace(shape, spacing=1.0, discretization=discretization)
    field = Element('m', values={'A': u + 1j * v, 'b': 1.0, 'c': -1.5}, space=lattice)
    joined = _system(
        field,
        lambda m, t: [RateLaw(m.A, m.A + (1 + I * m.b) * laplacian(m.A) - (1 + I * m.c) * abs(m.A) ** 2 * m.A)],
    )
    split = System(
        elements=[GinzburgLandauElement('m', u=u.ravel(), v=v.ravel(), space=lattice)],
        interactions=[GinzburgLandauInteraction('dynamics', target='m')],
    )
    return joined, split


def _constant(t, z):
    """Return a constant rate."""
    return 0.0


def test_harmonic_phase_turns_on_the_unit_circle():
    """Test that `dz/dt = 2 i z` reaches `exp(20 i)` at time 10 with modulus 1 at every frame."""
    system = phase_system(solver=ScipyODESolver(rtol=1e-12, atol=1e-12))
    trajectory = system.prepare().simulate(until=10, dt=0.5)
    values = trajectory.get_values('osc.z')
    assert values.dtype == np.complex128
    assert abs(values[-1] - np.exp(20j)) < 1e-8
    assert np.max(np.abs(np.abs(values) - 1)) < 1e-8


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_real_state_reads_modulus_of_complex_state(backend):
    """Test that a real state updated with the modulus of a complex state holds it after every step."""
    trajectory = _readout(backend=backend()).prepare().simulate(until=5, dt=1)
    assert trajectory.get_values('a.r').dtype == np.float64
    assert np.allclose(trajectory.get_values('a.r')[1:], 2.0)
    assert np.allclose(trajectory.get_values('a.z')[-1], 2 * np.exp(1.5j))


def test_complex_value_for_real_state_is_rejected():
    """Test that a complex value for a real state names the law, the state, and the ways out."""
    system = _system(Element('a', values={'x': 1.0}), lambda a, t: [RateLaw(a.x, I * a.x)])
    with pytest.raises(ValidationError, match=r'RateLaw of rule .* real state a\.x.*real_part.*imag_part.*domain=CC'):
        system.prepare()


def test_ordering_complex_value_is_rejected():
    """Test that a condition `z > 2` on a complex state suggests the modulus or the real part."""
    system = _system(Element('a', values={'z': 1.0 + 0j}), lambda a, t: [UpdateLaw(a.z, where(a.z > 2, a.z, 2 * a.z))])
    with pytest.raises(ValidationError, match=r'UpdateLaw of rule .*no order.*abs.*real_part'):
        system.prepare()


def test_complex_functions_split_into_parts():
    """Test that the parts, conjugate, argument, and a sum over a group give the values NumPy gives."""

    def parts(a, t):
        return [UpdateLaw(a.z, a.z), UpdateLaw(a.w, real_part(a.z) + I * imag_part(conjugate(a.z)) + arg(a.z))]

    problem = _system(Element('a', values={'z': 3 + 4j, 'w': 0j}), parts).prepare()._problem
    updated = problem.update(0.0, problem.initial_state, problem.initial_parameters)
    assert np.isclose(problem.states.read(updated, 'a.w'), 3 - 4j + np.angle(3 + 4j))
    values = np.array([0.0, 1.0, 2.0])
    group = Element('g', values={'z': values + 0j, 'm': np.zeros(3, dtype=complex)}, size=3)
    problem = _system(group, lambda g, t: [UpdateLaw(g.z, g.z), UpdateLaw(g.m, total(exp(I * g.z)))]).prepare()._problem
    updated = problem.update(0.0, problem.initial_state, problem.initial_parameters)
    assert np.allclose(problem.states.read(updated, 'g.m'), np.exp(1j * values).sum())


def test_backends_agree_on_complex_map():
    """Test that one step of `z^2 + c` on 10000 members agrees between NumPy and Numba to 1e-12."""
    rng = np.random.default_rng(0)
    z = rng.uniform(-1, 1, 10_000) + 1j * rng.uniform(-1, 1, 10_000)
    c = rng.uniform(-1, 1, 10_000) + 1j * rng.uniform(-1, 1, 10_000)
    points = QuadraticElement('points', z=z, c=c, size=10_000)
    reference, compiled = (
        quadratic_system(points, backend=NumpyBackend()).prepare()._problem,
        quadratic_system(points, backend=NumbaBackend()).prepare()._problem,
    )
    x, p = reference.initial_state, reference.initial_parameters
    expected, result = reference.update(0.0, x, p), compiled.update(0.0, x, p)
    assert compiled.backend == 'numba'
    assert np.allclose(result, expected, rtol=1e-12, atol=0)
    assert np.allclose(reference.states.read(expected, 'points.z'), z**2 + c, rtol=1e-12)


@pytest.mark.parametrize(
    'solver',
    [
        ScipyODESolver('RK23', rtol=1e-12, atol=1e-12),
        ScipyODESolver('RK45', rtol=1e-12, atol=1e-12),
        ScipyODESolver('DOP853', rtol=1e-12, atol=1e-12),
        ScipyODESolver('Radau', rtol=1e-12, atol=1e-12),
        ScipyODESolver('BDF', rtol=1e-12, atol=1e-12),
        ScipyODESolver('LSODA', rtol=1e-12, atol=1e-12),
        RK4Solver(dt=0.001),
    ],
    ids=lambda solver: getattr(solver, 'method', type(solver).__name__),
)
def test_ode_solvers_match_split_form(solver):
    """Test that every ODE solver runs a complex law that is not holomorphic as it runs the split real form."""
    joined = _stuart(solver=solver).prepare().simulate(until=5, dt=0.5)
    split = _split_stuart(solver=solver).prepare().simulate(until=5, dt=0.5)
    assert np.max(np.abs(joined.get_values('o.z') - (split.get_values('o.u') + 1j * split.get_values('o.v')))) < 1e-8


@pytest.mark.parametrize(
    ('solver', 'discretization'),
    [
        *((solver, discretization) for solver in STIFF for discretization in ('spectral', 'finite-difference')),
        (ETDRK4Solver(dt=0.1), 'spectral'),
    ],
)
def test_stiff_steppers_match_split_form(solver, discretization):
    """Test that the stiff steppers run a complex field as they run the split real form."""
    joined, split = _ginzburg_landau((16, 16), discretization)
    first = joined.set_params(solver=solver).prepare().simulate(until=5, dt=1.0)
    second = split.set_params(solver=solver).prepare().simulate(until=5, dt=1.0)
    assert np.max(np.abs(first.get_values('m.A') - (second.get_values('m.u') + 1j * second.get_values('m.v')))) < 1e-8


def test_complex_and_split_ginzburg_landau_agree():
    """Test that the complex and split Ginzburg-Landau equations agree to 1e-10 at every frame."""
    joined, split = _ginzburg_landau()
    first = joined.set_params(solver=ETDRK4Solver(dt=0.1)).prepare().simulate(until=20, dt=1.0)
    second = split.set_params(solver=ETDRK4Solver(dt=0.1)).prepare().simulate(until=20, dt=1.0)
    amplitude = first.get_lattice('m.A')
    assert amplitude.shape == (21, 32, 32)
    assert amplitude.dtype == np.complex128
    assert np.max(np.abs(amplitude - (second.get_lattice('m.u') + 1j * second.get_lattice('m.v')))) < 1e-10


def test_free_schrodinger_equation_keeps_the_norm():
    """Test that a Gaussian wave packet keeps its norm to 1e-8 relative error until time 5."""
    positions = np.arange(256) * 0.25
    packet = np.exp(-((positions - 32) ** 2) / 4 + 2j * positions)
    space = LatticeSpace((256,), spacing=0.25, discretization='spectral')
    wave = Element('wave', values={'psi': packet}, space=space)
    system = _system(wave, lambda w, t: [RateLaw(w.psi, I / 2 * laplacian(w.psi))], solver=ETDRK4Solver(dt=0.01))
    trajectory = system.prepare().simulate(until=5, dt=0.5)
    norms = (np.abs(trajectory.get_values('wave.psi')) ** 2).sum(axis=1)
    assert np.max(np.abs(norms / norms[0] - 1)) < 1e-8


@pytest.mark.skip(reason='Engine gap: the derivative z.diff(t) of a complex state is not read as its order.')
def test_complex_equation_and_event_split_into_parts():
    """Test that a second-order complex equation and an event on a complex state run on their parts."""

    def laws(o, t):
        return [ConstraintLaw(o.z.diff(t, 2) == -o.z), EventLaw(real_part(o.z) - 2, {o.z: 0}, terminal=True)]

    oscillator = Element('o', values={'z': 1.0 + 0j, "z'": 1j})
    system = _system(oscillator, laws, solver=ScipyODESolver(rtol=1e-10, atol=1e-10))
    trajectory = system.prepare().simulate(until=3, dt=0.5)
    assert np.allclose(trajectory.get_values('o.z'), np.exp(1j * trajectory.times), atol=1e-7)


@pytest.mark.parametrize(
    ('law', 'message'),
    [
        (lambda s: [JumpLaw(1.0, {s.z: 1})], 'changes a complex state'),
        (lambda s: [RateLaw(s.z, CallSpec(_constant, (s.z,)))], 'calls a function with complex values'),
        (lambda s: [RateLaw(s.z, mod(s.z, 2))], 'only real values'),
        (lambda s: [RateLaw(s.z, gamma(s.z))], 'cannot write with real and imaginary parts'),
        (
            lambda s: [UpdateLaw(s.z, lambda left: left, neighborhood=LatticeNeighborhood(((-1,),)))],
            'of rule needs a LatticeSpace',
        ),
    ],
)
def test_unsupported_complex_laws_are_rejected(law, message):
    """Test that a law the compiler cannot split names what it takes."""
    system = _system(Element('a', values={'z': 1.0 + 0j}), lambda a, t: law(a))
    with pytest.raises(ValidationError, match=message):
        system.prepare()
