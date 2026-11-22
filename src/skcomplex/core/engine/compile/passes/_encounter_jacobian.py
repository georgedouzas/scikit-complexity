"""Drop the exact Jacobian of a problem with encounter laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from .._base import BasePass


class EncounterJacobianPass(BasePass):
    """A pass that drops the exact Jacobian of a problem with encounter laws.

    The pass derives no Jacobian of the rates of encounters, so a solver of such a problem estimates it if it needs one.

    Read more in the user guide, under Extend the library.

    See Also:
        JacobianPass: Derives the Jacobian of other problems.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.social.opinion_dynamics import KineticOpinionSystem
        >>> compilation = PassCompiler().prepare(KineticOpinionSystem()).run().compilation_
        >>> compilation.jacobian is None
        True
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
