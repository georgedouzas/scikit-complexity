"""Define the spec of a named variant of a system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ...specs._base_spec import BaseSpec
from ..systems import CompositeSystem


@dataclass(frozen=True)
class ScenarioSpec(BaseSpec):
    """A spec of a named variant of a system, such as an intervention.

    A scenario changes values before the run, changes parameters at given times, or replaces the system. Every scenario
    of an experiment runs with the same random streams per design row and replicate, so scenarios differ only through
    what they change.

    It appears in many fields. An epidemic compares early and late lockdowns, a climate model compares emission paths,
    and an economy compares tax policies.

    Args:
        name:
            The name of the scenario.

        overrides:
            A mapping from factory argument or dotted path to the value it takes before the run.

        schedule:
            The changes during the run, as `(time, dotted parameter path, value)`, applied from the first frame at or
            after their time.

        system:
            A system that replaces the one the factory builds, for structural changes, or `None`.

    Examples:
        >>> from skcomplex.core.components.specs import ScenarioSpec
        >>> ScenarioSpec('reduced', schedule=[(50.0, 'source.rate', 0.025)]).schedule
        [(50.0, 'source.rate', 0.025)]
    """

    name: str = 'baseline'
    overrides: Mapping[str, object] | None = None
    schedule: Sequence[tuple[float, str, object]] = ()
    system: CompositeSystem | None = None
