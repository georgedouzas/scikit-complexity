"""Test spectral fields, stiff steppers, finite volumes, finite elements, and incompressible flow."""

import math
import warnings

import numpy as np
import pytest

from skcomplex.core.base import SolverError, ValidationError
from skcomplex.core.compile import compile_system
from skcomplex.core.model import Element, Field, Flux, LatticeSpace, Mesh, Rate, System, inverse_laplacian, laplacian
from skcomplex.core.runtime import Frame, Simulation
from skcomplex.core.solvers import ETDRK4, IMEX2, SSPRK, CrankNicolson, IMEXEuler, SplitOperator
from skcomplex.core.viz import RerunObserver, resolve_visual
from skcomplex.models.biology.ecology import logistic_growth
from skcomplex.models.physics.fluids import (
    burgers,
    decaying_turbulence,
    kuramoto_sivashinsky,
    rayleigh_benard,
    shallow_water,
)
from skcomplex.models.physics.heat import harmonic, heat_l_shape
from skcomplex.models.physics.pattern_formation import swift_hohenberg

Heat = Element.define('Heat', states={'u': 0.0}, laws=lambda s: [Rate(s.u, laplacian(s.u))])


def _diffused(discretization):
    """Return a smooth periodic profile diffused on 64 points and its exact solution."""
    n = 64
    spacing = 2 * math.pi / n
    positions = np.arange(n) * spacing
    initial = np.sin(3 * positions) + 0.5 * np.cos(5 * positions)
    exact = np.exp(-9 * 0.1) * np.sin(3 * positions) + 0.5 * np.exp(-25 * 0.1) * np.cos(5 * positions)
    rod = Field('rod', Heat, LatticeSpace(n, spacing=spacing), values={'u': initial}, discretization=discretization)
    values = Simulation(System(components=[rod]), dt=0.1, solver=CrankNicolson(dt=1e-4)).run(until=0.1).state_
    return np.abs(values - exact).max()


def _energies(vorticity):
    """Return the kinetic energy of every frame of a vorticity field on a periodic square of side 2 pi."""
    n = vorticity.shape[1]
    waves = np.fft.fftfreq(n, d=1.0 / n)
    squared = waves[:, None] ** 2 + np.fft.rfftfreq(n, d=1.0 / n)[None, :] ** 2
    squared[0, 0] = 1.0
    weights = np.ones(squared.shape)
    weights[:, 1:] = 2
    modes = np.abs(np.fft.rfftn(vorticity, axes=(1, 2))) ** 2 / squared
    modes[:, 0, 0] = 0
    return 0.5 * (weights * modes).sum(axis=(1, 2))


def test_spectral_accuracy():
    """Test that spectral diffusion on 64 points is at least 1000 times more accurate than finite differences."""
    assert _diffused('spectral') * 1000 <= _diffused('finite-difference')


def test_kuramoto_sivashinsky_is_fourth_order():
    """Test that ETDRK4 is stable at steps 0.25 and 0.125 and its error shrinks at fourth order."""
    problem = compile_system(kuramoto_sivashinsky())
    reference = Simulation(problem, dt=1.0, solver=ETDRK4(dt=1 / 256)).run(until=1).state_
    errors = [
        np.abs(Simulation(problem, dt=1.0, solver=ETDRK4(dt=step)).run(until=1).state_ - reference).max()
        for step in (0.25, 0.125)
    ]
    assert errors[0] / errors[1] == pytest.approx(16, rel=0.1)
    for step in (0.25, 0.125):
        values = Simulation(problem, dt=10.0, solver=ETDRK4(dt=step)).run(until=100).state_
        assert np.isfinite(values).all()
        assert np.abs(values).max() < 5


@pytest.mark.parametrize('stepper', [IMEXEuler(dt=1e-3), IMEX2(dt=1e-3), CrankNicolson(dt=1e-3)])
def test_implicit_explicit_steppers_agree(stepper):
    """Test that the implicit-explicit steppers converge to the ETDRK4 solution of the Kuramoto-Sivashinsky equation."""
    problem = compile_system(kuramoto_sivashinsky())
    reference = Simulation(problem, dt=1.0, solver=ETDRK4(dt=1 / 64)).run(until=1).state_
    values = Simulation(problem, dt=1.0, solver=stepper).run(until=1).state_
    tolerance = 1e-2 if isinstance(stepper, IMEXEuler) else 1e-5
    assert np.abs(values - reference).max() < tolerance


def test_split_operator_matches_linear_rates():
    """Test that the linear part of the rates equals the rates of a small perturbation of the coupled convection."""
    problem = compile_system(rayleigh_benard(1000.0, shape=(16, 16)))
    operator = SplitOperator.build(problem, problem.initial_parameters)
    noise = np.random.default_rng(0).standard_normal((2, 16, 16))
    state = 1e-7 * (noise - np.roll(np.flip(noise, axis=2), 1, axis=2)).ravel()
    rates = problem.drift(0.0, state, problem.initial_parameters)
    assert np.abs(rates - operator.apply(state)).max() < 1e-9 * np.abs(rates).max() + 1e-12


def test_steppers_need_linear_terms():
    """Test that split steppers reject problems without linear terms of fields, and ETDRK4 finite differences."""
    with pytest.raises(SolverError, match='no linear terms'):
        Simulation(logistic_growth(), solver=IMEXEuler()).run(until=1)
    with pytest.raises(SolverError, match='spectral'):
        Simulation(System(components=[Field('rod', Heat, LatticeSpace(8))]), dt=1, solver=ETDRK4()).run(until=1)


def test_burgers_shock_is_conservative():
    """Test that the Burgers shock keeps its total within 1e-12 and forms no new extrema."""
    trajectory = Simulation(burgers(), dt=0.25).run(until=2.0).trajectory_
    values = trajectory['line.u']
    totals = values.sum(axis=1)
    assert np.abs(totals - totals[0]).max() < 1e-12 * totals[0]
    assert values[-1].min() >= values[0].min() - 1e-12
    assert values[-1].max() <= values[0].max() + 1e-12
    jumps = np.abs(np.diff(values[-1]))
    assert jumps.max() > 0.5


@pytest.mark.parametrize('riemann', ['rusanov', 'godunov'])
def test_numerical_fluxes_conserve(riemann):
    """Test that the Rusanov and Godunov fluxes conserve the total of a shock without overshoots."""
    Shock = Element.define('Shock', states={'u': 0.0}, laws=lambda s: [Flux(s.u, s.u**2 / 2, riemann=riemann)])
    initial = 1.5 + np.sin((np.arange(100) + 0.5) * 2 * math.pi / 100)
    line = Field(
        'line',
        Shock,
        LatticeSpace(100, spacing=2 * math.pi / 100),
        values={'u': initial},
        discretization='finite-volume',
    )
    values = Simulation(System(components=[line]), dt=2.0, solver=SSPRK()).run(until=2.0).state_
    assert values.sum() == pytest.approx(initial.sum(), rel=1e-12)
    assert initial.min() - 1e-12 <= values.min() <= values.max() <= initial.max() + 1e-12


def test_dam_break_conserves_water():
    """Test that the dam break conserves its water and keeps the depth between the two initial depths."""
    values = Simulation(shallow_water(), dt=0.05).run(until=0.5).trajectory_['channel.h']
    assert values[-1].sum() == pytest.approx(values[0].sum(), rel=1e-12)
    assert 1.0 - 1e-9 <= values[-1].min() <= values[-1].max() <= 2.0 + 1e-9


def test_fluxes_need_finite_volume_field():
    """Test that flux laws on a finite-difference field are rejected."""
    Shock = Element.define('Shock', states={'u': 0.0}, laws=lambda s: [Flux(s.u, s.u**2 / 2)])
    with pytest.raises(ValidationError, match='finite-volume'):
        compile_system(System(components=[Field('line', Shock, LatticeSpace(10))]))


def test_l_shaped_steady_state_converges_at_second_order():
    """Test that heat on the L-shaped mesh converges at second order away from the re-entrant corner."""
    errors = []
    for refine in (2, 3, 4, 5):
        simulation = Simulation(heat_l_shape(refine), dt=100, solver=IMEXEuler(dt=10.0)).run(until=500)
        nodes = Mesh.l_shape(refine).nodes
        away = np.hypot(nodes[:, 0], nodes[:, 1]) > 0.25
        errors.append(np.abs(simulation.state_ - harmonic(nodes[:, 0], nodes[:, 1]))[away].max())
    ratios = np.asarray(errors[:-1]) / np.asarray(errors[1:])
    assert ratios[-1] == pytest.approx(4, rel=0.1)
    assert errors[-1] < 1e-4


def test_mesh_fields_stream_as_meshes(tmp_path):
    """Test that a finite element field logs a colored triangle mesh."""

    class Meshes:
        def __init__(self):
            self.logged = []

        def log(self, path, archetype):
            self.logged.append((path, type(archetype).__name__))

    simulation = Simulation(heat_l_shape(1), dt=0.1).run(until=0.1)
    element = simulation.problem_.elements[0]
    recording = Meshes()
    frame = Frame(0, 0.0, simulation.state_, simulation.parameters_, simulation.problem_)
    resolve_visual(element.visual).log(recording, element, frame)
    assert recording.logged == [('plate/u', 'Mesh3D')]
    observer = RerunObserver('mesh', save_path=tmp_path / 'mesh.rrd')
    Simulation(heat_l_shape(1), dt=0.1, observers=[observer]).run(until=0.2)
    assert observer.logged_frames_ == 3


def test_decaying_turbulence_loses_energy():
    """Test that decaying turbulence loses energy monotonically and enstrophy faster than energy."""
    vorticity = Simulation(decaying_turbulence(), dt=1.0).run(until=20).trajectory_.lattice('flow.omega')
    energy = _energies(vorticity)
    enstrophy = 0.5 * (vorticity**2).sum(axis=(1, 2))
    assert np.all(np.diff(energy) < 0)
    assert np.all(np.diff(enstrophy) < 0)
    assert enstrophy[-1] / enstrophy[0] < energy[-1] / energy[0]


def test_convection_onset():
    """Test that perturbations decay at Rayleigh number 500 and grow into steady rolls at 1000."""
    amplitudes = {}
    for rayleigh in (500.0, 1000.0):
        trajectory = Simulation(rayleigh_benard(rayleigh, shape=(16, 16)), dt=1.0).run(until=10).trajectory_
        amplitudes[rayleigh] = np.abs(trajectory['flow.theta']).max(axis=1)
    assert amplitudes[500.0][-1] < 1e-6
    assert amplitudes[1000.0][-1] > 0.1
    assert amplitudes[1000.0][-1] == pytest.approx(amplitudes[1000.0][-2], rel=1e-3)
    assert 500 < 27 * math.pi**4 / 4 < 1000


def test_symmetry_needs_periodic_lattice():
    """Test that parities on a lattice with fixed edges are rejected."""
    rod = Field('rod', Heat, LatticeSpace(8, boundary='fixed'), symmetry={'u': ('odd',)})
    with pytest.raises(ValidationError, match='periodic lattice'):
        compile_system(System(components=[rod]))


def test_poisson_needs_spectral_field():
    """Test that solving a Poisson equation on a finite-difference field is rejected."""
    Potential = Element.define('Potential', states={'u': 0.0}, laws=lambda s: [Rate(s.u, inverse_laplacian(s.u))])
    with pytest.raises(ValidationError, match='spectral field'):
        compile_system(System(components=[Field('rod', Potential, LatticeSpace(8))]))


def test_swift_hohenberg_rolls():
    """Test that Swift-Hohenberg above onset forms rolls with a wavelength within 5% of 2 pi."""
    simulation = Simulation(swift_hohenberg(), dt=10.0).run(until=200)
    values = simulation.trajectory_.lattice('cells.u')[-1]
    n, length = values.shape[0], 16 * math.pi
    spectrum = np.abs(np.fft.fft2(values)) ** 2
    spectrum[0, 0] = 0
    waves = 2 * math.pi * np.fft.fftfreq(n, length / n)
    peak = np.hypot(waves[:, None], waves[None, :]).ravel()[np.argmax(spectrum)]
    assert 2 * math.pi / peak == pytest.approx(2 * math.pi, rel=0.05)
    assert np.abs(values).max() > 0.5


def test_imex2_second_call_restarts_history():
    """Test that IMEX2 gives the same state from the same start twice in a row, as a fresh stepper does."""
    problem = compile_system(kuramoto_sivashinsky())
    x, p = problem.initial_state, problem.initial_parameters
    stepper, fresh = IMEX2(dt=0.01), IMEX2(dt=0.01)
    stepper.initialize(problem, np.random.default_rng(0))
    fresh.initialize(problem, np.random.default_rng(0))
    first = stepper.advance(0.0, x, p, 0.1)
    second = stepper.advance(0.0, x, p, 0.1)
    assert np.array_equal(first, second)
    assert np.array_equal(second, fresh.advance(0.0, x, p, 0.1))


def _viscous_burgers_laws(s):
    """Return the Burgers flux and its viscous Laplacian."""
    return [Flux(s.u, s.u**2 / 2), Rate(s.u, s.nu * laplacian(s.u))]


def test_viscous_burgers_respects_diffusive_limit():
    """Test that SSPRK keeps viscous Burgers within its initial range with no stability warning."""
    n = 200
    spacing = 2 * math.pi / n
    initial = 1.5 + np.sin((np.arange(n) + 0.5) * spacing)
    Viscous = Element.define('Viscous', states={'u': 0.0}, parameters={'nu': 0.5}, laws=_viscous_burgers_laws)
    space = LatticeSpace(n, spacing=spacing)
    field = Field('line', Viscous, space, values={'u': initial}, discretization='finite-volume')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        values = Simulation(System(components=[field]), solver=SSPRK(), dt=0.05).run(until=0.5).trajectory_['line.u']
    assert values.min() >= initial.min() - 1e-12
    assert values.max() <= initial.max() + 1e-12
