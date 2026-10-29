"""Derive the exact Jacobian of the drift of an ODE problem from its rate kernels."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.laws import RateLaw
from ....types import FormKind
from ...problems._jacobian_kernel import JacobianKernel
from .._base import BasePass
from .._compilation import Compilation


class JacobianPass(BasePass):
    """Pass that derives the exact Jacobian of the drift of an ODE problem, and none for other forms."""

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
            ).check()
            if compilation.form == FormKind.ODE
            else None
        )
