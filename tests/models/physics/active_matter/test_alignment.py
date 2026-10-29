"""Test the `_alignment` module of `skcomplex.models.physics.active_matter`."""

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import EulerMaruyamaSolver
from skcomplex.models import describe
from skcomplex.models.physics.active_matter import aligning_active_brownian_particles

from .conftest import _stationary_polarization


def test_aligning_active_brownian_particles_alignment_transition() -> None:
    """Polarization stays below 0.2 at weak alignment and exceeds 0.6 at strong alignment, at one density and noise."""
    polarizations = []
    for gamma in (0.02, 1.0):
        system = aligning_active_brownian_particles(200, size=20.0, gamma=gamma, D_r=0.1, random_state=0)
        simulation = (
            Simulation(
                system,
                dt=1.0,
                solver=EulerMaruyamaSolver(dt=0.05),
                random_state=0,
                observables=describe('aligning_active_brownian_particles').observables_of(system),
            )
            .check()
            .prepare()
            .run(until=200)
        )
        polarizations.append(_stationary_polarization(simulation.trajectory_, 100))
    low, high = polarizations
    assert low < 0.2
    assert high > 0.6
