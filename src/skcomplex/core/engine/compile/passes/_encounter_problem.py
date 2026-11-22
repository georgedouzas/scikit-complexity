"""Make the problem of a system with encounter laws a kinetic problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import fields

from ...problems._kinetic_problem_spec import KineticProblemSpec
from .._base import BasePass
from .._compilation import Compilation


class EncounterProblemPass(BasePass):
    """A pass that makes the problem of a system with encounter laws kinetic.

    The kinetic problem holds the encounter laws beside the rates, for a solver of densities of classes.

    Read more in the user guide, under Extend the library.

    See Also:
        CompileEncountersPass: Compiles the encounters.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.social.opinion_dynamics import KineticOpinionSystem
        >>> compilation = PassCompiler().prepare(KineticOpinionSystem()).run().compilation_
        >>> type(compilation.problem).__name__
        'KineticProblemSpec'
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
