"""Share the fixtures of the tests of the visuals."""

from collections.abc import Callable, Sequence
from dataclasses import replace

import pytest
import rerun as rr

from skcomplex.core.components.spaces import BaseSpace
from skcomplex.core.engine.compile import compile_system
from skcomplex.core.engine.problems import FrameSpec, ProblemSpec
from skcomplex.core.engine.viz import BaseVisual
from skcomplex.core.specs import AgentSnapshotSpec
from skcomplex.systems import create_system


@pytest.fixture(scope='module')
def log_agents() -> Callable[..., BaseVisual]:
    """Return a function that prepares a visual for agents in a space and logs one frame per snapshot."""
    problem = compile_system(create_system('moran_process'))
    assert isinstance(problem, ProblemSpec)

    def log(
        visual: BaseVisual,
        recording: rr.RecordingStream | None,
        path: str,
        snapshots: Sequence[AgentSnapshotSpec],
        space: BaseSpace | None = None,
    ) -> BaseVisual:
        """Fit a visual to agents with the traits of the first snapshot, log every snapshot, and return the visual."""
        values = dict(snapshots[0].values)
        agents = replace(problem.agents[0], path=path, states=tuple(values), values=values)
        prepared = replace(problem, agent_space=space, agents=(agents,))
        visual.set_params(path=path, recording=recording).prepare(prepared)
        for step, snapshot in enumerate(snapshots):
            frame = FrameSpec(
                step,
                float(step),
                prepared.initial_state,
                prepared.initial_parameters,
                prepared,
                {path: snapshot},
            )
            visual.run(frame)
        return visual

    return log
