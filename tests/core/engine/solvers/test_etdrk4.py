"""Test the `ETDRK4Solver` class."""

import math
from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.laws import FluxLaw, RateLaw
from skcomplex.core.components.laws.utils import laplacian
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.problems import FrameSpec
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import (
    CrankNicolsonSolver,
    ETDRK4Solver,
    IMEX2Solver,
    IMEXEulerSolver,
)
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.errors import SolverError, ValidationError
from skcomplex.core.specs import StateSpec, VisualSpec
from skcomplex.systems.biology.ecology import LogisticGrowthSystem
from skcomplex.systems.physics.fluids import (
    BurgersSystem,
    DecayingTurbulenceSystem,
    KuramotoSivashinskySystem,
    RayleighBenardSystem,
    ShallowWaterSystem,
)
from skcomplex.systems.physics.heat import HeatLShapeSystem
from skcomplex.systems.physics.pattern_formation import SwiftHohenbergSystem


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
    values = simulate(CompositeSystem(elements=[rod]), until=0.1, dt=0.1, solver=CrankNicolsonSolver(dt=1e-4)).states[
        -1
    ]
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
    system = KuramotoSivashinskySystem().prepare().system_
    reference = simulate(system, until=1, dt=1.0, solver=ETDRK4Solver(dt=1 / 256)).states[-1]
    errors = [
        np.abs(
            simulate(system, until=1, dt=1.0, solver=ETDRK4Solver(dt=step)).states[-1] - reference,
        ).max()
        for step in (0.25, 0.125)
    ]
    assert errors[0] / errors[1] == pytest.approx(16, rel=0.1)
    for step in (0.25, 0.125):
        values = simulate(system, until=100, dt=10.0, solver=ETDRK4Solver(dt=step)).states[-1]
        assert np.isfinite(values).all()
        assert np.abs(values).max() < 5


@pytest.mark.parametrize('stepper', [IMEXEulerSolver(dt=1e-3), IMEX2Solver(dt=1e-3), CrankNicolsonSolver(dt=1e-3)])
def test_implicit_explicit_steppers_agree(stepper):
    """Test that the implicit-explicit steppers converge to the ETDRK4 solution of the Kuramoto-Sivashinsky equation."""
    system = KuramotoSivashinskySystem().prepare().system_
    reference = simulate(system, until=1, dt=1.0, solver=ETDRK4Solver(dt=1 / 64)).states[-1]
    values = simulate(system, until=1, dt=1.0, solver=stepper).states[-1]
    tolerance = 1e-2 if isinstance(stepper, IMEXEulerSolver) else 1e-5
    assert np.abs(values - reference).max() < tolerance


def test_steppers_need_linear_terms():
    """Test that split steppers reject problems without linear terms of fields, and ETDRK4 finite differences."""
    with pytest.raises(SolverError, match='no linear terms'):
        simulate(LogisticGrowthSystem().prepare().system_, until=1, solver=IMEXEulerSolver())
    with pytest.raises(SolverError, match='spectral'):
        simulate(
            CompositeSystem(elements=[FieldElement('rod', element=HeatElement, space=LatticeSpace(8))]),
            until=1,
            dt=1,
            solver=ETDRK4Solver(),
        )


def test_burgers_shock_is_conservative():
    """Test that the Burgers shock keeps its total within 1e-12 and forms no new extrema."""
    trajectory = simulate(BurgersSystem().prepare().system_, until=2.0, dt=0.25)
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
    values = simulate(ShallowWaterSystem().prepare().system_, until=0.5, dt=0.05).values('channel.h')
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

    system = CompositeSystem(elements=[FieldElement('line', element=ShockElement, space=LatticeSpace(10))])
    with pytest.raises(ValidationError, match='finite-volume'):
        PassCompiler().prepare(system).run()


def test_mesh_fields_stream_as_meshes(tmp_path):
    """Test that a finite element field logs a colored triangle mesh."""

    class Meshes:
        def __init__(self):
            self.logged = []

        def log(self, path, archetype):
            self.logged.append((path, type(archetype).__name__))

    trajectory = simulate(HeatLShapeSystem(1).prepare().system_, until=0.1, dt=0.1)
    problem, recording = trajectory.problem, Meshes()
    visual = RerunObserver().prepare(problem).visuals_[0].set_params(recording=recording).prepare(problem)
    visual.run(FrameSpec(0, 0.0, trajectory.states[-1], trajectory.parameters[-1], problem))
    assert recording.logged == [('plate/u', 'Mesh3D')]
    observer = RerunObserver('mesh', save_path=tmp_path / 'mesh.rrd')
    simulate(HeatLShapeSystem(1).prepare().system_, until=0.2, dt=0.1, observers=[observer])
    assert observer.logged_frames_ == 3


@pytest.mark.slow
def test_decaying_turbulence_loses_energy():
    """Test that decaying turbulence loses energy monotonically and enstrophy faster than energy."""
    vorticity = simulate(DecayingTurbulenceSystem().prepare().system_, until=20, dt=1.0).lattice('flow.omega')
    energy = _energies(vorticity)
    enstrophy = 0.5 * (vorticity**2).sum(axis=(1, 2))
    assert np.all(np.diff(energy) < 0)
    assert np.all(np.diff(enstrophy) < 0)
    assert enstrophy[-1] / enstrophy[0] < energy[-1] / energy[0]


def test_convection_onset():
    """Test that perturbations decay at Rayleigh number 500 and grow into steady rolls at 1000."""
    amplitudes = {}
    for rayleigh in (500.0, 1000.0):
        trajectory = simulate(
            RayleighBenardSystem(rayleigh, shape=(16, 16)).prepare().system_,
            until=10,
            dt=1.0,
            solver=ETDRK4Solver(dt=0.01),
        )
        amplitudes[rayleigh] = np.abs(trajectory.values('flow.theta')).max(axis=1)
    assert amplitudes[500.0][-1] < 1e-6
    assert amplitudes[1000.0][-1] > 0.1
    assert amplitudes[1000.0][-1] == pytest.approx(amplitudes[1000.0][-2], rel=1e-3)
    assert 500 < 27 * math.pi**4 / 4 < 1000


def test_symmetry_needs_periodic_lattice():
    """Test that parities on a lattice with fixed edges are rejected."""
    rod = FieldElement('rod', element=HeatElement, space=LatticeSpace(8, boundary='fixed'), symmetry={'u': ('odd',)})
    with pytest.raises(ValidationError, match='periodic lattice'):
        PassCompiler().prepare(CompositeSystem(elements=[rod])).run()


def test_swift_hohenberg_rolls():
    """Test that Swift-Hohenberg above onset forms rolls with a wavelength within 5% of 2 pi."""
    trajectory = simulate(SwiftHohenbergSystem().prepare().system_, until=200, dt=10.0, solver=ETDRK4Solver(dt=0.5))
    values = trajectory.lattice('cells.u')[-1]
    n, length = values.shape[0], 16 * math.pi
    spectrum = np.abs(np.fft.fft2(values)) ** 2
    spectrum[0, 0] = 0
    waves = 2 * math.pi * np.fft.fftfreq(n, length / n)
    peak = np.hypot(waves[:, None], waves[None, :]).ravel()[np.argmax(spectrum)]
    assert 2 * math.pi / peak == pytest.approx(2 * math.pi, rel=0.05)
    assert np.abs(values).max() > 0.5
