"""Read the random draws of update laws from values drawn after the parameters at every step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from .._base import BasePass
from .._compilation import Compilation
from .._draw_lowering import DrawLowering


class DrawsPass(BasePass):
    """Pass that replaces the draws of update laws by values drawn after the parameters."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('drawn', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('drawn', 'draws')

    def _run(self, inputs: object = None) -> None:
        """Lower the draws of the laws."""
        compilation = self.compilation_
        drawing = DrawLowering().check().prepare(compilation)
        drawn = drawing.run(Compilation._required(compilation.drawn, 'drawn'))
        if not isinstance(drawn, list):
            error_msg = f'The draw lowering returns a {type(drawn).__name__}, not a list of laws.'
            raise TypeError(error_msg)
        compilation.drawn = drawn
        drawing._apply_references()
        compilation.draws = drawing.draws_
