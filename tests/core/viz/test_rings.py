"""Test the visual of phases against positions of oscillator rings."""

import numpy as np

from skcomplex.core.runtime import Frame, Simulation
from skcomplex.core.viz import PhaseRingVisual, default_blueprint
from skcomplex.models import create


class Logs:
    """Recording stand-in that keeps the entity paths and archetypes it receives."""

    def __init__(self):
        """Start with no logs."""
        self.logs = []

    def log(self, path, archetype, static=False):
        """Keep a logged entity path and archetype."""
        self.logs.append((path, archetype))

    def of(self, path):
        """Return the archetypes logged under a path."""
        return [archetype for logged, archetype in self.logs if logged == path]


def test_phase_ring_visual_logs_wrapped_phases_against_index():
    """Test that every oscillator of a chimera ring is a point at its index and its phase in (-π, π], drawn upward."""
    simulation = Simulation(create('nonlocal_ring_chimera', n_oscillators=20, R=3)).run(until=0.0)
    problem = simulation.problem_
    frame = Frame(0, 0.0, simulation.state_, simulation.parameters_, problem)
    logs = Logs()
    PhaseRingVisual('theta', position='x').log(logs, problem.elements[0], frame)
    points = np.asarray(logs.of('ring/phases')[0].positions.as_arrow_array().to_pylist())
    assert points.shape == (20, 2)
    np.testing.assert_allclose(points[:, 0], np.arange(20))
    phases = -points[:, 1] / (0.6 * 19 / (2 * np.pi))
    assert np.all((phases > -np.pi - 1e-9) & (phases <= np.pi + 1e-9))


def test_chimera_ring_shows_phases_first():
    """Test that the default blueprint of the nonlocal ring chimera shows the phases of the ring first."""
    blueprint = default_blueprint(create('nonlocal_ring_chimera', n_oscillators=20, R=3))
    stack, views = [blueprint.root_container], []
    while stack:
        part = stack.pop(0)
        if hasattr(part, 'origin'):
            views.append(part)
        else:
            stack = list(part.contents) + stack
    assert str(views[0].origin) == 'ring/phases'
