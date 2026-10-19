"""Define scenarios, named overrides and scheduled changes that experiments compare."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from typing import Any

from sklearn.base import BaseEstimator

from ..model import System


class Scenario(BaseEstimator):
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
        >>> from skcomplex.experiments import Scenario
        >>> Scenario('quota', schedule=[(50.0, 'harvest.effort', 0.025)]).schedule
        [(50.0, 'harvest.effort', 0.025)]
    """

    def __init__(
        self,
        name: str = 'baseline',
        overrides: Mapping[str, Any] | None = None,
        schedule: Sequence[tuple[float, str, Any]] = (),
        system: System | None = None,
    ) -> None:
        self.name = name
        self.overrides = overrides
        self.schedule = schedule
        self.system = system
