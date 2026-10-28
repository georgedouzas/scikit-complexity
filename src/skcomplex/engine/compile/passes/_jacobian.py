"""Derive the exact Jacobian of the drift of an ODE problem from its rate kernels."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from ....core.laws import RateLaw
from ...problems import Form, Lowered, ReductionSchedule
from ...problems._jacobian_kernel import JacobianKernel
from .._compilation import Compilation
from .._declarations import Declarations
from .._pipeline import BasePass


class JacobianPass(BasePass):
    """Pass that derives the exact Jacobian of the drift of an ODE problem, and none for other forms."""

    reads = ('kernels', 'form', 'reductions', 'layout')
    writes = ('jacobian',)

    def _run(self, compilation: Compilation) -> None:
        """Derive the Jacobian of the drift.

        Args:
            compilation:
                The compilation state.
        """
        declarations = cast('Declarations', compilation.layout)
        compilation.jacobian = (
            JacobianKernel(
                tuple(cast('Lowered', compilation.kernels).kernels[RateLaw]),
                cast('ReductionSchedule', compilation.reductions),
                declarations.ends[True],
                declarations.time,
            ).check()
            if compilation.form == Form.ODE
            else None
        )
