"""Read the random draws of update laws from values drawn after the parameters at every step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from .._base import BasePass
from .._compilation import Laws
from .._draw_lowering import DrawLowering


class DrawsPass(BasePass):
    """Pass that replaces the draws of update laws by values drawn after the parameters."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('drawn', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('drawn', 'drawing')

    def _run(self, inputs: object = None) -> None:
        """Lower the draws of the laws."""
        compilation = self.compilation_
        drawing = DrawLowering().check().prepare(compilation)
        compilation.drawn = drawing.run(cast('Laws', compilation.drawn))
        compilation.drawing = drawing
