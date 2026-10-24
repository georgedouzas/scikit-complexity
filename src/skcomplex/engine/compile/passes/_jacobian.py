"""Derive the exact Jacobian of the drift of an ODE problem from its rate kernels."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from ....core.base import Form
from ....core.laws import Rate
from .._compilation import Compilation
from .._declarations import Declarations
from .._jacobians import ExactJacobian
from .._kernels import Lowered
from .._pipeline import Pass
from .._reductions import ReductionPass


class JacobianPass(Pass):
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
                tuple(cast('Lowered', compilation.kernels).kernels[Rate]),
                cast('ReductionPass', compilation.reductions),
                declarations.ends[True],
                declarations.time,
            )
            if compilation.form is Form.ODE
            else None
        )
