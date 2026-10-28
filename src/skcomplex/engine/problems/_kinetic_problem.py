"""Problem whose drift holds the gain and loss rates of encounter laws, with the compiled encounters."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

from ._encounter_kernel import EncounterKernel
from ._problem import Problem


@dataclass(frozen=True)
class KineticProblem(Problem):
    """Problem whose drift holds the gain and loss rates of encounter laws, with the compiled encounters.

    Args:
        encounters:
            The encounter laws, whose gain and loss rates the drift holds and particle solvers sample.
    """

    encounters: tuple[EncounterKernel, ...] = field(default=(), repr=False)
