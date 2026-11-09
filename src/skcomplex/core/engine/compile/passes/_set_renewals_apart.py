"""Pass that takes the renewal laws out of the laws lowered into kernels.

The renewal laws compile on their own.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from .._base import BasePass
from .._compilation import Compilation
from .._renewal_law import RenewalLaw


class SetRenewalsApartPass(BasePass):
    """Pass that takes the renewal laws out of the laws lowered into kernels.

    The renewal laws compile on their own.
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
