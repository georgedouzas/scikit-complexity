"""Problem whose drift holds the gain and loss rates of encounter laws, with the compiled encounters."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

from ...problems import Problem
from .._compiled_encounter import CompiledEncounter


@dataclass(frozen=True)
class KineticProblem(Problem):
    """Problem whose drift holds the gain and loss rates of encounter laws, with the compiled encounters.

    Args:
        encounters:
            The encounter laws, whose gain and loss rates the drift holds and particle solvers sample.
    """

    encounters: tuple[CompiledEncounter, ...] = field(default=(), repr=False)
