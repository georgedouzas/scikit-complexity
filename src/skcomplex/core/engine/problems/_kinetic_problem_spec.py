"""Define the kinetic problem spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass, field

from ._encounter_kernel import EncounterKernel
from ._problem_spec import ProblemSpec


@dataclass(frozen=True)
class KineticProblemSpec(ProblemSpec):
    """A spec of a problem whose rates come from encounters between classes.

    It is a problem of rate laws whose rates hold the gains and losses of encounters, such as agents of two opinions who
    meet and change opinion, and it also keeps the compiled encounters, so a particle solver can sample them one by one
    instead of integrating the densities. It models opinions in a population, wealth among traders, and molecules of a
    gas.

    Read more in the user guide, under Extend the library.

    Args:
        encounters:
            The encounter laws, whose gain and loss rates the drift holds and particle solvers sample.

    See Also:
        EncounterKernel: One compiled encounter.
        ProblemSpec: The problem it extends.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import KineticOpinionSystem
        >>> problem = KineticOpinionSystem().prepare().simulate(until=0).problem
        >>> type(problem).__name__, len(problem.encounters)
        ('KineticProblemSpec', 1)
    """

    encounters: tuple[EncounterKernel, ...] = field(default=(), repr=False)
