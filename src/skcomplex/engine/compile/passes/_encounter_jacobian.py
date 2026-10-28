"""Pass that drops the exact Jacobian of a problem with encounter laws, whose rates it does not derive."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from .._compilation import Compilation
from .._pipeline import BasePass


class EncounterJacobianPass(BasePass):
    """Pass that drops the exact Jacobian of a problem with encounter laws, whose rates it does not derive."""

    reads = ('jacobian', 'encounters')
    writes = ('jacobian',)

    def _run(self, compilation: Compilation) -> None:
        """Drop the Jacobian of a problem with encounter laws.

        Args:
            compilation:
                The compilation state.
        """
        if compilation.facts['encounters']:
            compilation.jacobian = None
