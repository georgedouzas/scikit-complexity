"""Define the spec of a named variant of a system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..base import BaseSpec
from .types import AnySystem


@dataclass(frozen=True)
class ScenarioSpec(BaseSpec):
    """A spec of a named variant of a system, such as an intervention.

    It changes values before the run, changes parameters at given times, or replaces the system, such as
    `ScenarioSpec('distancing', overrides={'beta': 0.0001})` for a lower rate of infection. Every scenario of an
    experiment runs with the same random streams, so scenarios differ only through what they change. An epidemic
    compares early and late lockdowns, a climate model compares paths of emissions, and an economy compares tax
    policies.

    Read more in the user guide, under Run experiments.

    Args:
        name:
            The name of the scenario.

        overrides:
            A mapping from a name `set_params` takes, such as `beta` or `growth__r`, or a dotted path, to the value it
            takes before the run.

        schedule:
            The changes during the run, as `(time, dotted parameter path, value)`, applied from the first frame at or
            after their time.

        system:
            A system that replaces the one the factory builds, for structural changes, or `None`.

    See Also:
        compare_scenarios: Runs the scenarios of a system.

    Examples:
        >>> from skcomplex.core.experiments import ScenarioSpec
        >>> ScenarioSpec('reduced', schedule=[(50.0, 'source.rate', 0.025)]).schedule
        [(50.0, 'source.rate', 0.025)]
    """

    name: str = 'baseline'
    overrides: Mapping[str, object] | None = None
    schedule: Sequence[tuple[float, str, object]] = ()
    system: AnySystem | None = None
