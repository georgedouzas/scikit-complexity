"""Rewrite the random draws of update laws as numbers drawn at every step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from .._base import BasePass
from .._compilation import Compilation
from .._draw_lowering import DrawLowering


class DrawsPass(BasePass):
    """A pass that rewrites the random draws of update laws as numbers drawn each step.

    An update law may draw a random number, such as a random turn for every bird of a flock. The pass rewrites every
    draw as a read of a number that the run draws before every step and stores after the parameters.

    Read more in the user guide, under Extend the library.

    See Also:
        DrawLowering: Rewrites the draws.
        DrawKernel: Draws the numbers before every step.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.physics.active_matter import VicsekSystem
        >>> compilation = PassCompiler().prepare(VicsekSystem()).run().compilation_
        >>> compilation.draws.size
        400
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('drawn', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('drawn', 'draws')

    def _run(self, inputs: object = None) -> None:
        """Lower the draws of the laws."""
        compilation = self.compilation_
        drawing = DrawLowering().prepare(compilation)
        compilation.drawn = drawing.run(Compilation._get_required(compilation.drawn, 'drawn')).lowered_
        drawing._apply_references()
        compilation.draws = drawing.draws_
