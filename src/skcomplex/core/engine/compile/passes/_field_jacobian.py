"""Pass that adds the conservation laws, symmetries, and fixed nodes of the fields to the exact Jacobian."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....errors import PipelineError
from ...problems._jacobian_kernel import JacobianKernel
from .._base import BasePass


class FieldJacobianPass(BasePass):
    """Pass that adds the conservation laws, symmetries, and fixed nodes of the fields to the exact Jacobian."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('jacobian', 'volumes', 'symmetries', 'constrained')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('jacobian',)

    def _run(self, inputs: object = None) -> None:
        """Add the discretizations of the fields to the Jacobian."""
        compilation = self.compilation_
        if compilation.jacobian is None:
            return
        if compilation.constrained is None:
            error_msg = 'FieldJacobianPass reads the fixed nodes of the fields, which no pass wrote.'
            raise PipelineError(error_msg)
        settings = compilation.jacobian.get_params(deep=False)
        compilation.jacobian = JacobianKernel(
            **settings
            | {
                'volumes': compilation.volumes,
                'symmetries': compilation.symmetries,
                'constrained': compilation.constrained[0],
            },
        )._validated()
