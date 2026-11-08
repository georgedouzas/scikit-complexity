"""Test the visual of phases against positions of oscillator rings."""

import numpy as np

from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.problems import FrameSpec
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import PhaseRingVisual, RerunObserver
from skcomplex.systems.physics.synchronization import NonlocalRingChimeraSystem


class Logs:
    """Recording stand-in that keeps the entity paths and archetypes it receives."""

    def __init__(self):
        self.logs = []

    def log(self, path, archetype, static=False):
        """Keep a logged entity path and archetype."""
        self.logs.append((path, archetype))

    def of(self, path):
        """Return the archetypes logged under a path."""
        return [archetype for logged, archetype in self.logs if logged == path]


def test_phase_ring_visual_logs_wrapped_phases_against_index():
    """Test that every oscillator of a chimera ring is a point at its index and its phase in (-π, π], drawn upward."""
    trajectory = simulate(NonlocalRingChimeraSystem(n_oscillators=20, R=3).prepare().system_, until=0.0)
    problem = trajectory.problem
    frame = FrameSpec(0, 0.0, trajectory.states[-1], trajectory.parameters[-1], problem)
    logs = Logs()
    (PhaseRingVisual('theta', position='x', path=problem.elements[0].path, recording=logs).prepare(problem).run(frame))
    points = np.asarray(logs.of('ring/phases')[0].positions.as_arrow_array().to_pylist())
    assert points.shape == (20, 2)
    np.testing.assert_allclose(points[:, 0], np.arange(20))
    phases = -points[:, 1] / (0.6 * 19 / (2 * np.pi))
    assert np.all((phases > -np.pi - 1e-9) & (phases <= np.pi + 1e-9))


def test_chimera_ring_shows_phases_first():
    """Test that the default blueprint of the nonlocal ring chimera shows the phases of the ring first."""
    problem = PassCompiler().prepare(NonlocalRingChimeraSystem(n_oscillators=20, R=3).prepare().system_).run().problem_
    blueprint = RerunObserver(blueprint='default').prepare(problem).blueprint_
    stack, views = [blueprint.root_container], []
    while stack:
        part = stack.pop(0)
        if hasattr(part, 'origin'):
            views.append(part)
        else:
            stack = list(part.contents) + stack
    assert str(views[0].origin) == 'ring/phases'
