"""Derive the exact Jacobian of the drift of an ODE problem from its rate kernels."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from ....core.laws import RateLaw
from ....specs import DeclarationsSpec, LoweredKernelsSpec, ReductionScheduleSpec
from ....types import FormKind
from ...problems._jacobian_kernel import JacobianKernel
from .._base import BasePass


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
        declarations = cast('DeclarationsSpec', compilation.layout)
        compilation.jacobian = (
            JacobianKernel(
                tuple(cast('LoweredKernelsSpec', compilation.kernels).kernels[RateLaw]),
                cast('ReductionScheduleSpec', compilation.reductions),
                declarations.ends[True],
                declarations.time,
            ).check()
            if compilation.form == FormKind.ODE
            else None
        )
