"""Share the fixtures of the tests of the visuals."""

from collections.abc import Callable, Sequence
from dataclasses import replace

import pytest
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.components.spaces import BaseSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.problems import BaseProblemSpec, FrameSpec, ProblemSpec
from skcomplex.core.engine.runtime import BaseObserver, simulate
from skcomplex.core.engine.viz import BaseVisual, RerunObserver
from skcomplex.core.specs import AgentSnapshotSpec
from skcomplex.systems.biology.evolution import MoranProcessSystem


class Logs:
    """Recording stand-in that keeps the entity paths and archetypes it receives."""

    def __init__(self) -> None:
        self.logs: list[tuple[str, object]] = []
        self.static: list[str] = []

    def log(self, path: str, archetype: object, static: bool = False) -> None:
        """Keep a logged entity path and archetype, and whether it is static."""
        self.logs.append((path, archetype))
        if static:
            self.static.append(path)

    def of(self, path: str) -> list[object]:
        """Return the archetypes logged under a path."""
        return [archetype for logged, archetype in self.logs if logged == path]


def _frames(system: CompositeSystem, steps: int, dt: float = 0.1) -> tuple[BaseProblemSpec, list[FrameSpec]]:
    """Return the problem and the frames of a run of a system."""
    frames = []

    class KeepObserver(BaseObserver):
        """Observer that keeps every frame."""

        def _run(self, inputs: FrameSpec | None) -> None:
            """Keep a frame."""
            if inputs is not None:
                frames.append(inputs)

    trajectory = simulate(system, until=steps * dt, dt=dt, observers=[KeepObserver()], random_state=0)
    return trajectory.problem, frames


def _first_frame(system: CompositeSystem) -> tuple[BaseProblemSpec, FrameSpec]:
    """Return the problem and the first frame of a system."""
    trajectory = simulate(system, until=0.0)
    problem = trajectory.problem
    return problem, FrameSpec(0, 0.0, trajectory.states[-1], trajectory.parameters[-1], problem)


def _radius(archetype: rr.Points2D | rr.Points3D) -> float:
    """Return the first radius of logged points."""
    return archetype.radii.as_arrow_array().to_pylist()[0]


def _prepared(system: CompositeSystem | BaseProblemSpec, **settings: object) -> RerunObserver:
    """Return a Rerun observer prepared on a system or a compiled problem."""
    problem = system if isinstance(system, BaseProblemSpec) else PassCompiler().prepare(system).run().problem_
    return RerunObserver(**settings).prepare(problem)


def default_blueprint(system: CompositeSystem | BaseProblemSpec, **settings: object) -> rrb.Blueprint:
    """Return the default blueprint of a system or a compiled problem."""
    return _prepared(system, blueprint='default', **settings).blueprint_


def _views(blueprint: rrb.Blueprint) -> list[rrb.View]:
    """Return the views of a blueprint in layout order."""
    found = []
    stack = [blueprint.root_container]
    while stack:
        part = stack.pop(0)
        if isinstance(part, rrb.View):
            found.append(part)
        else:
            stack = list(part.contents) + stack
    return found


@pytest.fixture(scope='module')
def log_agents() -> Callable[..., BaseVisual]:
    """Return a function that prepares a visual for agents in a space and logs one frame per snapshot."""
    problem = PassCompiler().prepare(MoranProcessSystem().prepare().system_).run().problem_
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
