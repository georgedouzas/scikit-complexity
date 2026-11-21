"""Derive the exact Jacobian of a problem of rates from its kernels."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....components.laws import RateLaw
from ....enums import FormKind
from ...problems._jacobian_kernel import JacobianKernel
from .._base import BasePass
from .._compilation import Compilation


class JacobianPass(BasePass):
    """A pass that derives the exact Jacobian of a problem of rates.

    The Jacobian holds how fast every rate changes with every state, which implicit solvers of stiff problems use. The
    pass derives it exactly from the laws of a problem of rates or of a static problem, and derives none for others.

    Read more in the user guide, under Extend the library.

    See Also:
        FieldJacobianPass: Adds the terms of fields.
        EncounterJacobianPass: Drops it for encounters.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.biology.ecology import LotkaVolterraSystem
        >>> compilation = PassCompiler().prepare(LotkaVolterraSystem()).run().compilation_
        >>> compilation.jacobian.size
        2
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
        declarations = Compilation._get_required(compilation.layout, 'layout')
        compilation.jacobian = (
            JacobianKernel(
                tuple(Compilation._get_required(compilation.kernels, 'kernels').kernels[RateLaw]),
                Compilation._get_required(compilation.reductions, 'reductions'),
                declarations.ends[True],
                declarations.time,
            )
            if compilation.form in (FormKind.ODE, FormKind.STATIC)
            else None
        )
