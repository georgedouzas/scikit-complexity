"""Pass that adds the conservation laws, symmetries, and fixed nodes of the fields to the exact Jacobian."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace

from .._compilation import Compilation
from .._pipeline import BasePass


class FieldJacobianPass(BasePass):
    """Pass that adds the conservation laws, symmetries, and fixed nodes of the fields to the exact Jacobian."""

    reads = ('jacobian', 'volumes', 'symmetries', 'constrained')
    writes = ('jacobian',)

    def _run(self, compilation: Compilation) -> None:
        """Add the discretizations of the fields to the Jacobian.

        Args:
            compilation:
                The compilation state.
        """
        if compilation.jacobian is None:
            return
        compilation.jacobian = replace(
            compilation.jacobian,
            volumes=compilation.facts['volumes'],
            symmetries=compilation.facts['symmetries'],
            constrained=compilation.facts['constrained'][0],
        )
