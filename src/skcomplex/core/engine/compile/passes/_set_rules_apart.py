"""Set the rules of agents apart from the laws that kernels evaluate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....components.laws._rule import _RuleLaw
from .._base import BasePass
from .._compilation import Compilation


class SetRulesApartPass(BasePass):
    """A pass that sets the rules of agents apart from the laws that kernels evaluate.

    A rule is a Python function that changes agents, such as a trader who gives a coin, so the agent solver calls it
    in place of a compiled kernel.

    Read more in the user guide, under Extend the library.

    See Also:
        AgentsProblemPass: Adds the rules to the problem.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.economics.econophysics import BoltzmannWealthSystem
        >>> compilation = PassCompiler().prepare(BoltzmannWealthSystem()).run().compilation_
        >>> [(spec.owner, len(rules)) for spec, rules in compilation.rules]
        [('traders', 0), ('gifts', 1)]
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('reduced',)

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('reduced', 'rules')

    def _run(self, inputs: object = None) -> None:
        """Take the rules out of the reduced laws."""
        compilation = self.compilation_
        reduced = Compilation._get_required(compilation.reduced, 'reduced')
        compilation.rules = [(spec, tuple(law for law in laws if isinstance(law, _RuleLaw))) for spec, laws in reduced]
        compilation.reduced = [(spec, [law for law in laws if not isinstance(law, _RuleLaw)]) for spec, laws in reduced]
