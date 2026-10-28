"""Define the kinetic problem spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ._problem_spec import ProblemSpec

if TYPE_CHECKING:
    from ..engine.problems._encounter_kernel import EncounterKernel


@dataclass(frozen=True)
class KineticProblemSpec(ProblemSpec):
    """Problem whose drift holds the gain and loss rates of encounter laws, with the compiled encounters.

    Args:
        encounters:
            The encounter laws, whose gain and loss rates the drift holds and particle solvers sample.
    """

    encounters: tuple[EncounterKernel, ...] = field(default=(), repr=False)
