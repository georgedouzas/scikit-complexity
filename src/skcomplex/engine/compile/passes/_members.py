"""Lower the member offset reads of laws to neighbour reads of the index table of their group."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from .._base import BasePass
from .._compilation import Laws
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
        compilation.shifted = [
            (spec, MemberLowering(spec).check().prepare(compilation).run(laws))
            for spec, laws in cast('Laws', compilation.corrected)
        ]
