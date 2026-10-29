"""Define the fitness spec of the Brock-Hommes traders."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ....specs import BaseSpec


@dataclass(frozen=True)
class _FitnessSpec(BaseSpec):
    """Fitness of every heuristic at one time, which the traders read once per step and keep in the storage.

    Args:
        time:
            The time the fitness was read at.

        fitness:
            The last fitness of every heuristic.
    """

    time: float
    fitness: tuple[float, ...]
