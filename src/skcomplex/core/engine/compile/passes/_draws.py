"""Read the random draws of update laws from values drawn after the parameters.

The values are drawn at every step.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from .._base import BasePass
from .._compilation import Compilation
from .._draw_lowering import DrawLowering


class DrawsPass(BasePass):
    """Pass that replaces the draws of update laws by values drawn.

    The values are drawn after the parameters.
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
        compilation.drawn = drawing.run(Compilation._required(compilation.drawn, 'drawn')).lowered_
        drawing._apply_references()
        compilation.draws = drawing.draws_
