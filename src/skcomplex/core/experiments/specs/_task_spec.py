"""Define the spec of one run of an experiment, sent to a worker process."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ...components.specs import ScenarioSpec
from ...engine.runtime import Trajectory
from ...engine.solvers import BaseSolver
from ...specs._base_spec import BaseSpec
from ...specs._observable_spec import ObservableSpec
from ..types import Output, Row, Source


@dataclass(frozen=True)
class _TaskSpec(BaseSpec):
    """One run of an experiment, sent as it is to a worker process.

    Args:
        index:
            The position of the run in the experiment.

        source:
            The system, or the factory of systems.

        row:
            The values of the run, by model parameter or dotted path.

        scenario:
            The scenario.

        replicate:
            The replicate number.

        seed:
            The random stream of the run.

        until:
            The end time of the run, or `None` for the tuned end time of the system.

        dt:
            The time between frames, or `None` for the tuned one.

        solver:
            The solver, or `None` for the tuned one.

        outputs:
            The outputs to keep, or `None` for every observable or state.

        warmup:
            The simulated time before which kept series are discarded.

        observables:
            The observables of the run, or none for the tuned ones.

        analyses:
            The analyses of the run, functions of its trajectory, by their column prefix.
    """

    index: int
    source: Source
    row: Row
    scenario: ScenarioSpec
    replicate: int
    seed: np.random.SeedSequence
    until: float | None
    dt: float | None
    solver: BaseSolver | None
    outputs: Sequence[Output] | None
    warmup: float
    observables: Sequence[ObservableSpec]
    analyses: tuple[tuple[str, Callable[[Trajectory], object]], ...]
