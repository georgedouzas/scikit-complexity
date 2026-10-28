"""Pass that drops the exact Jacobian of a problem with encounter laws, whose rates it does not derive."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from .._base import BasePass


class EncounterJacobianPass(BasePass):
    """Pass that drops the exact Jacobian of a problem with encounter laws, whose rates it does not derive."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('jacobian', 'encounters')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('jacobian',)

    def _run(self, inputs: object = None) -> None:
        """Drop the Jacobian of a problem with encounter laws."""
        compilation = self.compilation_
        if compilation.encounters:
            compilation.jacobian = None
