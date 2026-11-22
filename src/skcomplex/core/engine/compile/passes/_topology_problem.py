"""Add the changes of the shape of networks to the problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace

from .._base import BasePass
from .._compilation import Compilation


class TopologyProblemPass(BasePass):
    """A pass that adds the changes of the shape of networks to the problem.

    The run then applies them, such as the newcomer of a growing network and the links it makes.

    Read more in the user guide, under Extend the library.

    See Also:
        TopologyFunctionsPass: Compiles the changes.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.mathematics.networks import BarabasiAlbertSystem
        >>> compilation = PassCompiler().prepare(BarabasiAlbertSystem()).run().compilation_
        >>> [request.owner for request in compilation.problem.topology.requests]
        ['growth']
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('problem', 'topology')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('problem',)

    def _run(self, inputs: object = None) -> None:
        """Add the topology changes to the problem."""
        compilation = self.compilation_
        problem = Compilation._get_required(compilation.problem, 'problem')
        compilation.problem = replace(problem, topology=compilation.topology)
