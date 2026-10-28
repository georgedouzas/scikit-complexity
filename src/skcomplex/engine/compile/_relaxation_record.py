"""Relaxation of a lattice within one step, such as an avalanche."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass


@dataclass(frozen=True)
class RelaxationRecord:
    """Relaxation of a lattice within one step, such as an avalanche.

    Args:
        label:
            The label of the relaxation.

        time:
            The time of the step.

        size:
            The number of site updates.

        duration:
            The number of relaxation iterations.

        area:
            The number of distinct sites that changed.
    """

    label: str
    time: float
    size: int
    duration: int
    area: int
