"""Test the `ETDRK4Solver` class."""

import math
from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element, FieldElement
from skcomplex.core.laws import FluxLaw, RateLaw
from skcomplex.core.laws.utils import inverse_laplacian, laplacian
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import (
    CrankNicolsonSolver,
    ETDRK4Solver,
    IMEX2Solver,
    IMEXEulerSolver,
)
from skcomplex.engine.viz import RerunObserver
from skcomplex.errors import SolverError, ValidationError
from skcomplex.models.biology.ecology import logistic_growth
from skcomplex.models.physics.fluids import (
    burgers,
    decaying_turbulence,
    kuramoto_sivashinsky,
    rayleigh_benard,
    shallow_water,
)
from skcomplex.models.physics.heat import heat_l_shape
from skcomplex.models.physics.pattern_formation import swift_hohenberg
from skcomplex.specs import FrameSpec, StateSpec, VisualSpec


class HeatElement(Element):
    """Heat element."""

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
        """Return the laws of the heat element."""
        return [RateLaw(s.u, laplacian(s.u))]


def _diffused(discretization):
    """Return a smooth periodic profile diffused on 64 points and its exact solution."""
    n = 64
    spacing = 2 * math.pi / n
    positions = np.arange(n) * spacing
    initial = np.sin(3 * positions) + 0.5 * np.cos(5 * positions)
    exact = np.exp(-9 * 0.1) * np.sin(3 * positions) + 0.5 * np.exp(-25 * 0.1) * np.cos(5 * positions)
    rod = FieldElement(
        'rod',
        element=HeatElement,
        space=LatticeSpace(n, spacing=spacing),
        values={'u': initial},
        discretization=discretization,
    )
    values = (
        Simulation(System(elements=[rod]), dt=0.1, solver=CrankNicolsonSolver(dt=1e-4))
        .check()
        .run(until=0.1)
        .state_
    )
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
    reference = Simulation(problem, dt=1.0, solver=ETDRK4Solver(dt=1 / 256)).check().run(until=1).state_
    errors = [
        np.abs(Simulation(problem, dt=1.0, solver=ETDRK4Solver(dt=step)).check().run(until=1).state_ - reference).max()
        for step in (0.25, 0.125)
    ]
    assert errors[0] / errors[1] == pytest.approx(16, rel=0.1)
    for step in (0.25, 0.125):
        values = Simulation(problem, dt=10.0, solver=ETDRK4Solver(dt=step)).check().run(until=100).state_
        assert np.isfinite(values).all()
        assert np.abs(values).max() < 5


@pytest.mark.parametrize('stepper', [IMEXEulerSolver(dt=1e-3), IMEX2Solver(dt=1e-3), CrankNicolsonSolver(dt=1e-3)])
def test_implicit_explicit_steppers_agree(stepper):
    """Test that the implicit-explicit steppers converge to the ETDRK4 solution of the Kuramoto-Sivashinsky equation."""
    problem = compile_system(kuramoto_sivashinsky())
    reference = Simulation(problem, dt=1.0, solver=ETDRK4Solver(dt=1 / 64)).check().run(until=1).state_
    values = Simulation(problem, dt=1.0, solver=stepper).check().run(until=1).state_
    tolerance = 1e-2 if isinstance(stepper, IMEXEulerSolver) else 1e-5
    assert np.abs(values - reference).max() < tolerance


def test_steppers_need_linear_terms():
    """Test that split steppers reject problems without linear terms of fields, and ETDRK4 finite differences."""
    with pytest.raises(SolverError, match='no linear terms'):
        Simulation(logistic_growth(), solver=IMEXEulerSolver()).check().run(until=1)
    with pytest.raises(SolverError, match='spectral'):
        Simulation(
            System(elements=[FieldElement('rod', element=HeatElement, space=LatticeSpace(8))]),
            dt=1,
            solver=ETDRK4Solver(),
        ).check().run(until=1)


def test_burgers_shock_is_conservative():
    """Test that the Burgers shock keeps its total within 1e-12 and forms no new extrema."""
    trajectory = Simulation(burgers(), dt=0.25).check().run(until=2.0).trajectory_
    values = trajectory.values('line.u')
    totals = values.sum(axis=1)
    assert np.abs(totals - totals[0]).max() < 1e-12 * totals[0]
    assert values[-1].min() >= values[0].min() - 1e-12
    assert values[-1].max() <= values[0].max() + 1e-12
    jumps = np.abs(np.diff(values[-1]))
    assert jumps.max() > 0.5


@pytest.mark.slow
def test_dam_break_conserves_water():
    """Test that the dam break conserves its water and keeps the depth between the two initial depths."""
    values = Simulation(shallow_water(), dt=0.05).check().run(until=0.5).trajectory_.values('channel.h')
    assert values[-1].sum() == pytest.approx(values[0].sum(), rel=1e-12)
    assert 1.0 - 1e-9 <= values[-1].min() <= values[-1].max() <= 2.0 + 1e-9


def test_fluxes_need_finite_volume_field():
    """Test that flux laws on a finite-difference field are rejected."""

    class ShockElement(Element):
        """Shock element."""

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
            """Return the laws of the shock element."""
            return [FluxLaw(s.u, s.u**2 / 2)]

    with pytest.raises(ValidationError, match='finite-volume'):
        compile_system(System(elements=[FieldElement('line', element=ShockElement, space=LatticeSpace(10))]))


def test_mesh_fields_stream_as_meshes(tmp_path):
    """Test that a finite element field logs a colored triangle mesh."""

    class Meshes:
        def __init__(self):
            self.logged = []

        def log(self, path, archetype):
            self.logged.append((path, type(archetype).__name__))

    simulation = Simulation(heat_l_shape(1), dt=0.1).check().run(until=0.1)
    problem, recording = simulation.problem_, Meshes()
    visual = (
        RerunObserver()
        .check()
        .prepare(problem)
        .visuals_[0]
        .set_params(recording=recording)
        .check()
        .prepare(problem)
    )
    visual.run(FrameSpec(0, 0.0, simulation.state_, simulation.parameters_, problem))
    assert recording.logged == [('plate/u', 'Mesh3D')]
    observer = RerunObserver('mesh', save_path=tmp_path / 'mesh.rrd')
    Simulation(heat_l_shape(1), dt=0.1, observers=[observer]).check().run(until=0.2)
    assert observer.logged_frames_ == 3


def test_decaying_turbulence_loses_energy():
    """Test that decaying turbulence loses energy monotonically and enstrophy faster than energy."""
    vorticity = Simulation(decaying_turbulence(), dt=1.0).check().run(until=20).trajectory_.lattice('flow.omega')
    energy = _energies(vorticity)
    enstrophy = 0.5 * (vorticity**2).sum(axis=(1, 2))
    assert np.all(np.diff(energy) < 0)
    assert np.all(np.diff(enstrophy) < 0)
    assert enstrophy[-1] / enstrophy[0] < energy[-1] / energy[0]


def test_convection_onset():
    """Test that perturbations decay at Rayleigh number 500 and grow into steady rolls at 1000."""
    amplitudes = {}
    for rayleigh in (500.0, 1000.0):
        simulation = Simulation(rayleigh_benard(rayleigh, shape=(16, 16)), dt=1.0, solver=ETDRK4Solver(dt=0.01)).check()
        trajectory = simulation.run(until=10).trajectory_
        amplitudes[rayleigh] = np.abs(trajectory.values('flow.theta')).max(axis=1)
    assert amplitudes[500.0][-1] < 1e-6
    assert amplitudes[1000.0][-1] > 0.1
    assert amplitudes[1000.0][-1] == pytest.approx(amplitudes[1000.0][-2], rel=1e-3)
    assert 500 < 27 * math.pi**4 / 4 < 1000


def test_symmetry_needs_periodic_lattice():
    """Test that parities on a lattice with fixed edges are rejected."""
    rod = FieldElement('rod', element=HeatElement, space=LatticeSpace(8, boundary='fixed'), symmetry={'u': ('odd',)})
    with pytest.raises(ValidationError, match='periodic lattice'):
        compile_system(System(elements=[rod]))


def test_poisson_needs_spectral_field():
    """Test that solving a Poisson equation on a finite-difference field is rejected."""

    class PotentialElement(Element):
        """Potential element."""

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
            """Return the laws of the potential element."""
            return [RateLaw(s.u, inverse_laplacian(s.u))]

    with pytest.raises(ValidationError, match='spectral field'):
        compile_system(System(elements=[FieldElement('rod', element=PotentialElement, space=LatticeSpace(8))]))


def test_swift_hohenberg_rolls():
    """Test that Swift-Hohenberg above onset forms rolls with a wavelength within 5% of 2 pi."""
    simulation = Simulation(swift_hohenberg(), dt=10.0, solver=ETDRK4Solver(dt=0.5)).check().run(until=200)
    values = simulation.trajectory_.lattice('cells.u')[-1]
    n, length = values.shape[0], 16 * math.pi
    spectrum = np.abs(np.fft.fft2(values)) ** 2
    spectrum[0, 0] = 0
    waves = 2 * math.pi * np.fft.fftfreq(n, length / n)
    peak = np.hypot(waves[:, None], waves[None, :]).ravel()[np.argmax(spectrum)]
    assert 2 * math.pi / peak == pytest.approx(2 * math.pi, rel=0.05)
    assert np.abs(values).max() > 0.5
