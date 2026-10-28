"""Define the task spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator

from ..types import Row
from ._base_spec import BaseSpec
from ._observable_spec import ObservableSpec
from ._scenario_spec import ScenarioSpec

if TYPE_CHECKING:
    from ..experiments._base import Source
    from ..experiments._experiment import Output


@dataclass(frozen=True)
class _TaskSpec(BaseSpec):
    """One run of an experiment, sent as it is to a worker process.

    Args:
        index:
            The position of the run in the experiment.

        source:
            The factory or system of the model.

        row:
            The design values.

        scenario:
            The scenario.

        replicate:
            The replicate number.

        seed:
            The random stream of the run.

        until:
            The simulated time to run to.

        settings:
            The `Simulation` arguments.

        outputs:
            The outputs to keep.

        warmup:
            The simulated time before which kept series are discarded.

        observables:
            The observables of the run, or `None` for none.

        analyses:
            The unprepared analyses of the run by their column prefix.
    """

    index: int
    source: Source
    row: Row
    scenario: ScenarioSpec
    replicate: int
    seed: np.random.SeedSequence
    until: float
    settings: Mapping[str, object]
    outputs: Sequence[Output] | None
    warmup: float
    observables: Sequence[ObservableSpec] | None
    analyses: tuple[tuple[str, BaseEstimator], ...]
