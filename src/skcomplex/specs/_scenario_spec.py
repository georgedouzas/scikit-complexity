"""Define the scenario spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._base_spec import BaseSpec

if TYPE_CHECKING:
    from ..core.system import System


@dataclass(frozen=True)
class ScenarioSpec(BaseSpec):
    """Named variant of a model: overrides before the run, parameter changes at given times, or another system.

    Every scenario of an experiment runs with the same random streams per design row and replicate, so scenarios differ
    only through what they change.

    Args:
        name:
            The name of the scenario.

        overrides:
            A mapping from factory argument or dotted path to the value it takes before the run.

        schedule:
            The changes during the run, as `(time, dotted parameter path, value)`, applied from the first frame at or
            after their time.

        system:
            A system that replaces the model's system, for structural changes, or `None`.

    Examples:
        >>> from skcomplex.specs import ScenarioSpec
        >>> ScenarioSpec('quota', schedule=[(50.0, 'harvest.effort', 0.025)]).schedule
        [(50.0, 'harvest.effort', 0.025)]
    """

    name: str = 'baseline'
    overrides: Mapping[str, object] | None = None
    schedule: Sequence[tuple[float, str, object]] = ()
    system: System | None = None
