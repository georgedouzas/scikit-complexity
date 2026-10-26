"""Share the fixtures of the tests of the visuals."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import pytest

from skcomplex.core.problems import Problem
from skcomplex.core.space import BaseSpace
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Frame
from skcomplex.engine.solvers import AgentSnapshot
from skcomplex.engine.viz import BaseVisual
from skcomplex.models import create


@pytest.fixture(scope='module')
def log_agents() -> Callable[..., BaseVisual]:
    """Return a function that prepares a visual for agents in a space and logs one frame per snapshot."""
    problem = compile_system(create('moran_process'))
    assert isinstance(problem, Problem)

    def log(
        visual: BaseVisual,
        recording: Any,  # noqa: ANN401
        path: str,
        snapshots: Sequence[AgentSnapshot],
        space: BaseSpace | None = None,
    ) -> BaseVisual:
        """Fit a visual to agents with the traits of the first snapshot, log every snapshot, and return the visual."""
        prepared = replace(problem, agent_space=space)
        values = dict(snapshots[0].values)
        agents = replace(problem.agents[0], path=path, states=tuple(values), values=values)
        visual.prepare(agents, prepared)
        for step, snapshot in enumerate(snapshots):
            frame = Frame(
                step,
                float(step),
                prepared.initial_state,
                prepared.initial_parameters,
                prepared,
                {path: snapshot},
            )
            visual.run(frame, recording)
        return visual

    return log
