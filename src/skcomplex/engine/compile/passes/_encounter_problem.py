"""Pass that makes the problem of a system with encounter laws a kinetic problem that holds them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import fields
from typing import cast

from ...problems import Problem
from .._compilation import Compilation
from .._pipeline import BasePass
from ._kinetic_problem import KineticProblem


class EncounterProblemPass(BasePass):
    """Pass that makes the problem of a system with encounter laws a kinetic problem that holds them."""

    reads = ('problem', 'encounters')
    writes = ('problem',)

    def _run(self, compilation: Compilation) -> None:
        """Add the encounter laws to the problem.

        Args:
            compilation:
                The compilation state.
        """
        encounters = compilation.facts['encounters']
        if not encounters:
            return
        problem = cast('Problem', compilation.problem)
        values = {item.name: getattr(problem, item.name) for item in fields(problem)}
        compilation.problem = KineticProblem(**values, encounters=encounters)
