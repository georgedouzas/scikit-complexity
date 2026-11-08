"""Derive the exact Jacobian of an ODE or static problem from its kernels.

It is the Jacobian of the drift of an ODE problem, or of the residuals of a static one.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....components.laws import RateLaw
from ....types import FormKind
from ...problems._jacobian_kernel import JacobianKernel
from .._base import BasePass
from .._compilation import Compilation


class JacobianPass(BasePass):
    """Pass that derives the exact Jacobian of an ODE or static problem.

    It derives none for other forms.
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('kernels', 'form', 'reductions', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('jacobian',)

    def _run(self, inputs: object = None) -> None:
        """Derive the Jacobian of the drift."""
        compilation = self.compilation_
        declarations = Compilation._required(compilation.layout, 'layout')
        compilation.jacobian = (
            JacobianKernel(
                tuple(Compilation._required(compilation.kernels, 'kernels').kernels[RateLaw]),
                Compilation._required(compilation.reductions, 'reductions'),
                declarations.ends[True],
                declarations.time,
            )
            if compilation.form in (FormKind.ODE, FormKind.STATIC)
            else None
        )
