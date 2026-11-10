"""Define the kinetic problem spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass, field

from ._encounter_kernel import EncounterKernel
from ._problem_spec import ProblemSpec


@dataclass(frozen=True)
class KineticProblemSpec(ProblemSpec):
    """A spec of a problem whose drift holds the gain and loss rates of encounters.

    It holds the compiled encounters too.

    Args:
        encounters:
            The encounter laws, whose gain and loss rates the drift holds and particle solvers sample.
    """

    encounters: tuple[EncounterKernel, ...] = field(default=(), repr=False)
