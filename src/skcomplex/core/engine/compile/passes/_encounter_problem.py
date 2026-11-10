"""Pass that makes the problem of a system with encounter laws kinetic."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import fields

from ...problems._kinetic_problem_spec import KineticProblemSpec
from .._base import BasePass
from .._compilation import Compilation


class EncounterProblemPass(BasePass):
    """A pass that makes the problem of a system with encounter laws kinetic.

    The kinetic problem holds the encounter laws.
    """

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
        problem = Compilation._get_required(compilation.problem, 'problem')
        values = {item.name: getattr(problem, item.name) for item in fields(problem)}
        compilation.problem = KineticProblemSpec(**values, encounters=encounters)
