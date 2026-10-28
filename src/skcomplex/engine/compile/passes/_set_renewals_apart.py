"""Pass that takes the renewal laws out of the laws lowered into kernels, since they compile on their own."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from .._compilation import Compilation, Laws
from .._pipeline import BasePass
from .._renewal_law import RenewalLaw


class SetRenewalsApartPass(BasePass):
    """Pass that takes the renewal laws out of the laws lowered into kernels, since they compile on their own."""

    reads = ('reduced',)
    writes = ('reduced',)

    def _run(self, compilation: Compilation) -> None:
        """Take the renewal laws out of the reduced laws.

        Args:
            compilation:
                The compilation state.
        """
        compilation.reduced = [
            (spec, [law for law in laws if not isinstance(law, RenewalLaw)])
            for spec, laws in cast('Laws', compilation.reduced)
        ]
