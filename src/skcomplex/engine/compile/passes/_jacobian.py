"""Derive the exact Jacobian of the drift of an ODE problem from its rate kernels."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from ....core.base import Form
from ....core.laws import RateLaw
from ....core.problems import Lowered, ReductionSchedule
from .._compilation import Compilation
from .._declarations import Declarations
from .._jacobians import ExactJacobian
from .._pipeline import BasePass


class JacobianPass(BasePass):
    """Pass that derives the exact Jacobian of the drift of an ODE problem, and none for other forms."""

    reads = ('kernels', 'form', 'reductions', 'layout')
    writes = ('jacobian',)

    def run(self, compilation: Compilation) -> None:
        """Derive the Jacobian of the drift.

        Args:
            compilation:
                The compilation state.
        """
        declarations = cast('Declarations', compilation.layout)
        compilation.jacobian = (
            ExactJacobian(
                tuple(cast('Lowered', compilation.kernels).kernels[RateLaw]),
                cast('ReductionSchedule', compilation.reductions),
                declarations.ends[True],
                declarations.time,
            )
            if compilation.form == Form.ODE
            else None
        )
