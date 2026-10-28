"""Pass that makes the problem of a system with encounter laws a kinetic problem that holds them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import fields
from typing import cast

from ....specs import KineticProblemSpec, ProblemSpec
from .._base import BasePass


class EncounterProblemPass(BasePass):
    """Pass that makes the problem of a system with encounter laws a kinetic problem that holds them."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('problem', 'encounters')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('problem',)

    def _run(self, inputs: object = None) -> None:
        """Add the encounter laws to the problem."""
        compilation = self.compilation_
        encounters = compilation.encounters
        if not encounters:
            return
        problem = cast('ProblemSpec', compilation.problem)
        values = {item.name: getattr(problem, item.name) for item in fields(problem)}
        compilation.problem = KineticProblemSpec(**values, encounters=encounters)
