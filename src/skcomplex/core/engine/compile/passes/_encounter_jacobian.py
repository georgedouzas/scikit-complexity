"""Pass that drops the exact Jacobian of a problem with encounter laws.

It does not derive their rates.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from .._base import BasePass


class EncounterJacobianPass(BasePass):
    """Pass that drops the exact Jacobian of a problem with encounter laws.

    It does not derive their rates.
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('jacobian', 'encounters', 'volumes')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('jacobian',)

    def _run(self, inputs: object = None) -> None:
        """Drop the Jacobian of a problem with encounter laws or nonlocal fluxes.

        Nonlocal fluxes are fluxes that read nonlocal values.
        """
        compilation = self.compilation_
        nonlocal_fluxes = any(read is not None for volume in compilation.volumes or () for read in volume.reads)
        if compilation.encounters or nonlocal_fluxes:
            compilation.jacobian = None
