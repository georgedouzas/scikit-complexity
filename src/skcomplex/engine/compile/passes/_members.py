"""Lower the member offset reads of laws to neighbour reads of the index table of their group."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from .._base import BasePass
from .._compilation import Compilation
from .._members import MemberLowering


class MemberShiftPass(BasePass):
    """Pass that lowers the member offset reads of laws to neighbour reads."""

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
        for spec, laws in Compilation._required(compilation.corrected, 'corrected'):
            lowering = MemberLowering(spec).check().prepare(compilation)
            lowered = lowering.run(laws)
            if not isinstance(lowered, list):
                error_msg = f'The member lowering returns a {type(lowered).__name__}, not a list of laws.'
                raise TypeError(error_msg)
            compilation.shifted.append((spec, lowered))
            lowering._apply_references()
