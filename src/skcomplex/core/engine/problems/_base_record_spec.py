"""Define the base record spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...base import BaseSpec


@dataclass(frozen=True)
class BaseRecordSpec(BaseSpec):
    """An abstract spec of a record of something that happened during a run.

    A record holds what happened at one time of a run, such as an event, a cascade, or a renewal, with the readers
    `label`, the name of what happened, and `time`, when it happened. A run keeps them in `Trajectory.events`, such as
    the jumps of walkers, the bounces of a ball, and the avalanches of a sandpile.

    Read more in the user guide, under Analyse.

    See Also:
        Trajectory: Holds the records of a run in its events.

    Examples:
        >>> from skcomplex.core.engine.problems import BaseRecordSpec
        >>> from skcomplex.systems.physics.soft_matter import ContinuousTimeRandomWalkSystem
        >>> trajectory = ContinuousTimeRandomWalkSystem(n_walkers=3).prepare().simulate(until=10, random_state=0)
        >>> isinstance(trajectory.events[0], BaseRecordSpec), trajectory.events[0].label
        (True, 'jumps')
    """
