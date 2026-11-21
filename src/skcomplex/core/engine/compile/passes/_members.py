"""Rewrite reads of other members of a group by their offset."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from .._base import BasePass
from .._compilation import Compilation
from .._members import MemberLowering


class MemberShiftPass(BasePass):
    """A pass that rewrites reads of other members of a group by their offset.

    A law of a group may read the member next to the one it changes, such as the car ahead on a ring road. The pass
    rewrites every such read as a read of a symbol whose value comes from the member at that offset.

    Read more in the user guide, under Extend the library.

    See Also:
        MemberLowering: Rewrites the reads of one component.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.engineering.traffic import OptimalVelocitySystem
        >>> compilation = PassCompiler().prepare(OptimalVelocitySystem()).compilation_
        >>> reads = [str(law.value) for _, laws in compilation.shifted for law in laws]
        >>> 'x_cars__shift0' in reads[-1]
        True
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('corrected', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('shifted',)

    def _run(self, inputs: object = None) -> None:
        """Lower the member offset reads of the corrected laws."""
        compilation = self.compilation_
        compilation.shifted = []
        for spec, laws in Compilation._get_required(compilation.corrected, 'corrected'):
            lowering = MemberLowering(spec).prepare(compilation)
            compilation.shifted.append((spec, lowering.run(laws).lowered_))
            lowering._apply_references()
