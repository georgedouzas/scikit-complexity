"""Test complex states: their domain, storage, laws, backends, solvers, trajectories, and analysis guards."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import CC, NN, RR, I, arg, conjugate, exp, gamma, imag_part, oo, real_part
from sage.sets.real_set import RealSet

from skcomplex.core.analysis import spectrum
from skcomplex.core.components.elements import Element, FieldElement, GroupElement
from skcomplex.core.components.laws import CrossingLaw, EquationLaw, JumpLaw, LocalRuleLaw, RateLaw, RuleLaw, UpdateLaw
from skcomplex.core.components.laws.utils import derivative, laplacian, mod, total, where
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import (
    CrankNicolsonSolver,
    ETDRK4Solver,
    IMEX2Solver,
    IMEXEulerSolver,
    LammpsSolver,
    RK4Solver,
    ScipyODESolver,
)
from skcomplex.core.errors import BackendError, ValidationError
from skcomplex.core.specs import CallSpec, EntrySpec, LayoutSpec, ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from skcomplex.systems.physics.pattern_formation import GinzburgLandauElement

STIFF = (IMEXEulerSolver(dt=0.05), IMEX2Solver(dt=0.05), CrankNicolsonSolver(dt=0.05))


class PhaseElement(Element):
    """Complex amplitude that turns at a constant frequency."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        z: Annotated[complex, StateSpec(domain=CC)] = 1.0,
        omega: Annotated[float, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.z = z
        self.omega = omega

    def _get_laws(self, s):
        """Return `dz/dt = i omega z`."""
        return [RateLaw(s.z, I * s.omega * s.z)]


class QuadraticElement(Element):
    """Complex quadratic map `z -> z^2 + c`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        z: Annotated[complex, StateSpec(domain=CC)] = 0.0,
        c: Annotated[complex, ParameterSpec(domain=CC)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.z = z
        self.c = c

    def _get_laws(self, s):
        """Return the map."""
        return [UpdateLaw(s.z, s.z**2 + s.c)]


class ReadoutElement(Element):
    """Rotation of a complex state with its modulus read into a real state."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        z: Annotated[complex, StateSpec(domain=CC)] = 2.0,
        r: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.z = z
        self.r = r

    def _get_laws(self, s):
        """Return the rotation and the readout."""
        return [UpdateLaw(s.z, s.z * exp(I * 0.3)), UpdateLaw(s.r, abs(s.z))]


class CountElement(Element):
    """Amount that cannot be negative beside a count of whole events."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        amount: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        events: Annotated[float, StateSpec(domain=NN)] = 0,
        level: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.amount = amount
        self.events = events
        self.level = level

    def _get_laws(self, s):
        """Return constant states."""
        return [UpdateLaw(s.amount, s.amount), UpdateLaw(s.events, s.events), UpdateLaw(s.level, s.level)]


class StuartElement(Element):
    """Stuart-Landau oscillator, whose law is not holomorphic."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        z: Annotated[complex, StateSpec(domain=CC)] = 0.5 + 0.2j,
    ) -> None:
        super().__init__(label, visual=visual)
        self.z = z

    def _get_laws(self, s):
        """Return `dz/dt = (1 + i) z - |z|^2 z`."""
        return [RateLaw(s.z, (1 + I) * s.z - abs(s.z) ** 2 * s.z)]


class SplitStuartElement(Element):
    """Stuart-Landau oscillator written with its real and imaginary parts."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec()] = 0.5,
        v: Annotated[float, StateSpec()] = 0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.v = v

    def _get_laws(self, s):
        """Return the split law."""
        square = s.u**2 + s.v**2
        return [RateLaw(s.u, s.u - s.v - square * s.u), RateLaw(s.v, s.u + s.v - square * s.v)]


class ComplexGinzburgLandauElement(Element):
    """Complex Ginzburg-Landau equation with one complex field."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A: Annotated[complex, StateSpec(domain=CC)] = 0.0,
        b: Annotated[float, ParameterSpec()] = 1.0,
        c: Annotated[float, ParameterSpec()] = -1.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A = A
        self.b = b
        self.c = c

    def _get_laws(self, s):
        """Return `A + (1 + i b) laplacian(A) - (1 + i c) |A|^2 A`."""
        return [RateLaw(s.A, s.A + (1 + I * s.b) * laplacian(s.A) - (1 + I * s.c) * abs(s.A) ** 2 * s.A)]


class SchrodingerElement(Element):
    """Free Schrödinger equation."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        psi: Annotated[complex, StateSpec(domain=CC)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.psi = psi

    def _get_laws(self, s):
        """Return `dpsi/dt = i/2 laplacian(psi)`."""
        return [RateLaw(s.psi, I / 2 * laplacian(s.psi))]


def _ginzburg_landau(shape=(32, 32), discretization='spectral'):
    """Return the complex and split Ginzburg-Landau systems from one small random amplitude."""
    rng = np.random.default_rng(0)
    u, v = 0.01 * rng.uniform(-1, 1, shape), 0.01 * rng.uniform(-1, 1, shape)
    space = LatticeSpace(shape, spacing=1.0)
    values = {'A': u + 1j * v}
    joined = CompositeSystem(
        elements=[
            FieldElement(
                'm',
                element=ComplexGinzburgLandauElement,
                space=space,
                values=values,
                discretization=discretization,
            ),
        ],
    )
    split = CompositeSystem(
        elements=[
            FieldElement(
                'm',
                element=GinzburgLandauElement,
                space=space,
                values={'u': u, 'v': v},
                discretization=discretization,
            ),
        ],
    )
    return joined, split


def test_complex_state_reads_back_its_initial_value():
    """Test that a complex default is stored as a pair of floats and read back as a complex number."""
    problem = PassCompiler().prepare(CompositeSystem(elements=[PhaseElement('osc', z=1 + 2j)])).run().problem_
    assert problem.initial_state.tolist() == [1.0, 2.0]
    assert problem.states.read(problem.initial_state, 'osc.z') == 1 + 2j


def test_real_state_rejects_complex_value():
    """Test that a complex value of a real state names the state, its domain, and the complex domain."""
    with pytest.raises(ValidationError, match=r'osc\.omega = 1j.*its domain Real Field.*domain=CC'):
        PassCompiler().prepare(CompositeSystem(elements=[PhaseElement('osc', omega=1j)])).run()


@pytest.mark.parametrize(
    ('values', 'message'),
    [
        ({'amount': -1.0}, r'a\.amount = -1\.0 is not in its domain \[0, \+oo\)\.'),
        ({'events': 1.5}, r'a\.events = 1\.5 is not in its domain Non negative integer semiring\.'),
    ],
)
def test_real_value_outside_real_domain_is_reported_outside_it(values, message):
    """Test that a real value outside a real domain is reported as outside it, with no word of an imaginary part."""
    with pytest.raises(ValidationError, match=message) as error:
        PassCompiler().prepare(CompositeSystem(elements=[CountElement('a', **values)])).run()
    assert 'imaginary' not in str(error.value)


def test_complex_value_of_real_state_names_its_imaginary_part():
    """Test that a complex value of a real state is reported as having a nonzero imaginary part."""
    with pytest.raises(ValidationError, match=r'a\.level = \(1\+2j\) has a nonzero imaginary part'):
        PassCompiler().prepare(CompositeSystem(elements=[CountElement('a', level=1 + 2j)])).run()


def test_real_state_accepts_complex_value_with_zero_imaginary_part():
    """Test that a complex value with a zero imaginary part is a real value."""
    problem = PassCompiler().prepare(CompositeSystem(elements=[PhaseElement('osc', omega=3 + 0j)])).run().problem_
    assert problem.parameters.read(problem.initial_parameters, 'osc.omega') == 3.0


def test_complex_parameter_per_site_is_read_and_updated():
    """Test that every site reads its own complex parameter, and that an update changes the next step."""
    c = np.arange(16).reshape(4, 4) * (1 + 1j)
    grid = FieldElement('grid', element=QuadraticElement, space=LatticeSpace((4, 4)), values={'c': c})
    trajectory = simulate(CompositeSystem(elements=[grid]), until=1, dt=1)
    assert np.allclose(trajectory.get_lattice('grid.z')[1], c)
    grid.set_params(values={'c': np.full((4, 4), 2j)})
    trajectory = simulate(CompositeSystem(elements=[grid]), until=1, dt=1)
    assert np.allclose(trajectory.get_lattice('grid.z')[1], 2j)


def test_layout_of_real_systems_is_unchanged():
    """Test that a real variable keeps one float per component and its names."""
    entry = EntrySpec('a.x', 0, (2,), RR)
    assert (entry.size, entry.columns, entry.component_names) == (2, 2, ['a.x[0]', 'a.x[1]'])


def test_complex_view_is_zero_copy():
    """Test that the complex values of a layout view the stored floats without copying."""
    layout = LayoutSpec((EntrySpec('a.z', 0, (3,), CC, is_complex=True),))
    vector = np.arange(6.0)
    values = layout.read(vector, 'a.z')
    assert np.shares_memory(values, vector)
    assert values.tolist() == [1j, 2 + 3j, 4 + 5j]


def test_complex_component_is_located_by_its_part():
    """Test that the real and imaginary parts of a complex variable are located as real components."""
    layout = LayoutSpec((EntrySpec('a.z', 0, (), CC, is_complex=True),))
    entry, indices = layout.locate('a.z.imag')
    assert (entry.domain, indices) == (RR, slice(1, 2))


def test_harmonic_phase_turns_on_the_unit_circle():
    """Test that `dz/dt = 2 i z` reaches `exp(20 i)` at time 10 with modulus 1 at every frame."""
    trajectory = simulate(
        CompositeSystem(elements=[PhaseElement('osc')]),
        until=10,
        solver=ScipyODESolver(rtol=1e-12, atol=1e-12),
        dt=0.5,
    )
    values = trajectory.get_values('osc.z')
    assert values.dtype == np.complex128
    assert abs(values[-1] - np.exp(20j)) < 1e-8
    assert np.max(np.abs(np.abs(values) - 1)) < 1e-8


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_real_state_reads_modulus_of_complex_state(backend):
    """Test that a real state updated with the modulus of a complex state holds it after every step."""
    trajectory = simulate(CompositeSystem(elements=[ReadoutElement('a')]), until=5, backend=backend(), dt=1)
    assert trajectory.get_values('a.r').dtype == np.float64
    assert np.allclose(trajectory.get_values('a.r')[1:], 2.0)
    assert np.allclose(trajectory.get_values('a.z')[-1], 2 * np.exp(1.5j))


def test_complex_value_for_real_state_is_rejected():
    """Test that a complex value for a real state names the law, the state, and the ways out."""

    class WrongElement(Element):
        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[float, StateSpec()] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            return [RateLaw(s.x, I * s.x)]

    with pytest.raises(ValidationError, match=r'RateLaw of a .* real state a\.x.*real_part.*imag_part.*domain=CC'):
        PassCompiler().prepare(CompositeSystem(elements=[WrongElement('a')])).run()


def test_ordering_complex_value_is_rejected():
    """Test that a condition `z > 2` on a complex state suggests the modulus or the real part."""

    class OrderedElement(Element):
        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            z: Annotated[complex, StateSpec(domain=CC)] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.z = z

        def _get_laws(self, s):
            return [UpdateLaw(s.z, where(s.z > 2, s.z, 2 * s.z))]

    with pytest.raises(ValidationError, match=r'UpdateLaw of a .*no order.*abs.*real_part'):
        PassCompiler().prepare(CompositeSystem(elements=[OrderedElement('a')])).run()


def test_complex_functions_split_into_parts():
    """Test that the parts, conjugate, argument, and a sum over a group give the values NumPy gives."""

    class PartsElement(Element):
        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            z: Annotated[complex, StateSpec(domain=CC)] = 3 + 4j,
            w: Annotated[complex, StateSpec(domain=CC)] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.z = z
            self.w = w

        def _get_laws(self, s):
            return [UpdateLaw(s.z, s.z), UpdateLaw(s.w, real_part(s.z) + I * imag_part(conjugate(s.z)) + arg(s.z))]

    problem = PassCompiler().prepare(CompositeSystem(elements=[PartsElement('a')])).run().problem_
    updated = problem.update(0.0, problem.initial_state, problem.initial_parameters)
    assert np.isclose(problem.states.read(updated, 'a.w'), 3 - 4j + np.angle(3 + 4j))

    class MeanElement(Element):
        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            z: Annotated[complex, StateSpec(domain=CC)] = 0.0,
            m: Annotated[complex, StateSpec(domain=CC)] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.z = z
            self.m = m

        def _get_laws(self, s):
            return [UpdateLaw(s.z, s.z), UpdateLaw(s.m, total(exp(I * s.z)))]

    values = np.array([0.0, 1.0, 2.0])
    system = CompositeSystem(elements=[GroupElement('g', element=MeanElement, size=3, values={'z': values})])
    problem = PassCompiler().prepare(system).run().problem_
    updated = problem.update(0.0, problem.initial_state, problem.initial_parameters)
    assert np.allclose(problem.states.read(updated, 'g.m'), np.exp(1j * values).sum())


def test_backends_agree_on_complex_map():
    """Test that one step of `z^2 + c` on 10000 members agrees between NumPy and Numba to 1e-12."""
    rng = np.random.default_rng(0)
    z = rng.uniform(-1, 1, 10_000) + 1j * rng.uniform(-1, 1, 10_000)
    c = rng.uniform(-1, 1, 10_000) + 1j * rng.uniform(-1, 1, 10_000)
    system = CompositeSystem(
        elements=[GroupElement('points', element=QuadraticElement, size=10_000, values={'z': z, 'c': c})],
    )
    reference, compiled = (
        PassCompiler(backend=NumpyBackend()).prepare(system).run().problem_,
        PassCompiler(backend=NumbaBackend()).prepare(system).run().problem_,
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
    joined = simulate(CompositeSystem(elements=[StuartElement('o')]), until=5, solver=solver, dt=0.5)
    split = simulate(CompositeSystem(elements=[SplitStuartElement('o')]), until=5, solver=solver, dt=0.5)
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
    first = simulate(joined, until=5, solver=solver, dt=1.0)
    second = simulate(split, until=5, solver=solver, dt=1.0)
    assert np.max(np.abs(first.get_values('m.A') - (second.get_values('m.u') + 1j * second.get_values('m.v')))) < 1e-8


def test_complex_and_split_ginzburg_landau_agree():
    """Test that the complex and split Ginzburg-Landau equations agree to 1e-10 at every frame."""
    joined, split = _ginzburg_landau()
    first = simulate(joined, until=20, solver=ETDRK4Solver(dt=0.1), dt=1.0)
    second = simulate(split, until=20, solver=ETDRK4Solver(dt=0.1), dt=1.0)
    amplitude = first.get_lattice('m.A')
    assert amplitude.shape == (21, 32, 32)
    assert amplitude.dtype == np.complex128
    assert np.max(np.abs(amplitude - (second.get_lattice('m.u') + 1j * second.get_lattice('m.v')))) < 1e-10


def test_free_schrodinger_equation_keeps_the_norm():
    """Test that a Gaussian wave packet keeps its norm to 1e-8 relative error until time 5."""
    positions = np.arange(256) * 0.25
    packet = np.exp(-((positions - 32) ** 2) / 4 + 2j * positions)
    space = LatticeSpace((256,), spacing=0.25)
    system = CompositeSystem(
        elements=[
            FieldElement(
                'wave',
                element=SchrodingerElement,
                space=space,
                values={'psi': packet},
                discretization='spectral',
            ),
        ],
    )
    trajectory = simulate(system, until=5, solver=ETDRK4Solver(dt=0.01), dt=0.5)
    norms = (np.abs(trajectory.get_values('wave.psi')) ** 2).sum(axis=1)
    assert np.max(np.abs(norms / norms[0] - 1)) < 1e-8


def test_power_spectrum_rejects_complex_series():
    """Test that the power spectrum of a complex series names the real views to take first."""
    values = simulate(CompositeSystem(elements=[PhaseElement('osc')]), until=5, dt=0.1).get_values('osc.z')
    with pytest.raises(ValidationError, match=r'complex.*values\.real.*values\.imag.*np\.abs.*np\.angle'):
        spectrum(values)
    assert spectrum(values.real).power.size


def test_lammps_backend_rejects_complex_state():
    """Test that the LAMMPS backend names the complex state it cannot run."""
    with pytest.raises(BackendError, match=r'LAMMPS backend.*complex state osc\.z'):
        simulate(CompositeSystem(elements=[PhaseElement('osc')]), until=0.1, solver=LammpsSolver(), dt=0.1)


def test_mesa_backend_rejects_complex_agent_state():
    """Test that an agent type with a complex state names the state and the Mesa backend."""

    class WalkerElement(Element):
        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            z: Annotated[complex, StateSpec(domain=CC)] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.z = z

        def _get_laws(self, s):
            return [RuleLaw(lambda agent, system: None)]

    with pytest.raises(BackendError, match=r'walkers.*complex state z.*Mesa'):
        PassCompiler().prepare(CompositeSystem(elements=[GroupElement('walkers', element=WalkerElement, size=3)])).run()


def _constant(t, z):
    """Return a constant rate."""
    return 0.0


def _double(s):
    """Return twice the complex state of the oscillator."""
    return 2 * s.osc.z


def _size(s):
    """Return the modulus of the complex state of the oscillator."""
    return abs(s.osc.z)


def test_complex_equation_and_event_split_into_parts():
    """Test that a second-order complex equation and an event on a complex state run on their parts."""

    class OscillatorElement(Element):
        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            z: Annotated[complex, StateSpec(domain=CC, order=2, initial=(1j,))] = 1.0,
            z_d1: complex = 1j,
        ) -> None:
            super().__init__(label, visual=visual)
            self.z = z
            self.z_d1 = z_d1

        def _get_laws(self, s):
            return [EquationLaw(derivative(s.z, 2) == -s.z), CrossingLaw(real_part(s.z) - 2, {s.z: 0}, terminal=True)]

    solver = ScipyODESolver(rtol=1e-10, atol=1e-10)
    trajectory = simulate(CompositeSystem(elements=[OscillatorElement('o')]), until=3, solver=solver, dt=0.5)
    assert np.allclose(trajectory.get_values('o.z'), np.exp(1j * trajectory.times), atol=1e-7)


@pytest.mark.parametrize(
    ('law', 'message'),
    [
        (lambda s: [JumpLaw(1.0, {s.z: 1})], 'changes a complex state'),
        (lambda s: [RateLaw(s.z, CallSpec(_constant, (s.z,)))], 'calls a function with complex values'),
        (lambda s: [RateLaw(s.z, mod(s.z, 2))], 'only real values'),
        (lambda s: [RateLaw(s.z, gamma(s.z))], 'cannot write with real and imaginary parts'),
        (
            lambda s: [LocalRuleLaw(s.z, LatticeNeighborhood(((-1,),)), lambda left: left)],
            'LocalRuleLaw of a needs a LatticeSpace',
        ),
    ],
)
def test_unsupported_complex_laws_are_rejected(law, message):
    """Test that a law the compiler cannot split names what it takes."""

    class BadElement(Element):
        """Bad element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            z: Annotated[object, StateSpec(domain=CC)] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.z = z

        def _get_laws(self, s):
            """Return the laws of the bad element."""
            return law(s)

    bad = BadElement
    with pytest.raises(ValidationError, match=message):
        PassCompiler().prepare(CompositeSystem(elements=[bad('a')])).run()


def test_complex_observable_is_complex():
    """Test that an observable of a complex state is complex, and the observable of its modulus is real."""
    system = CompositeSystem(
        elements=[PhaseElement('osc', z=3 + 4j, omega=0.0)],
    )
    observables = [ObservableSpec('double', _double), ObservableSpec('size', _size)]
    trajectory = simulate(system, until=0.5, dt=0.5, observables=observables)
    assert trajectory.get_values('double')[0] == 6 + 8j
    assert trajectory.get_values('size')[0] == 5.0
