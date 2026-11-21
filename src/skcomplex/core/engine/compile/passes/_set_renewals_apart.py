"""Set the renewal laws apart from the laws that kernels evaluate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from .._base import BasePass
from .._compilation import Compilation
from .._renewal_law import RenewalLaw


class SetRenewalsApartPass(BasePass):
    """A pass that sets the renewal laws apart from the laws that kernels evaluate.

    A renewal law runs on a clock of its own member by member, so it compiles apart from the other laws.

    Read more in the user guide, under Extend the library.

    See Also:
        RenewalProblemPass: Compiles the renewal laws.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.physics.soft_matter import ContinuousTimeRandomWalkSystem
        >>> compilation = PassCompiler().prepare(ContinuousTimeRandomWalkSystem()).run().compilation_
        >>> [law for _, laws in compilation.reduced for law in laws]
        []
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('reduced',)

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('reduced',)

    def _run(self, inputs: object = None) -> None:
        """Take the renewal laws out of the reduced laws."""
        compilation = self.compilation_
        compilation.reduced = [
            (spec, [law for law in laws if not isinstance(law, RenewalLaw)])
            for spec, laws in Compilation._get_required(compilation.reduced, 'reduced')
        ]
