"""Read the random draws of update laws from values drawn after the parameters at every step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from .._compilation import Compilation, Laws
from .._declarations import Declarations
from .._pipeline import Pass
from .._stochastic import DrawLowering


class DrawsPass(Pass):
    """Pass that replaces the draws of update laws by values drawn after the parameters."""

    reads = ('drawn', 'layout')
    writes = ('drawn', 'drawing')

    def run(self, compilation: Compilation) -> None:
        """Lower the draws of the laws.

        Args:
            compilation:
                The compilation state.
        """
        declarations = cast('Declarations', compilation.layout)
        drawing = DrawLowering(declarations.ends[False], declarations.time)
        compilation.drawn = [
            (spec, drawing.lower(spec.owner, spec.rows, laws, spec.references))
            for spec, laws in cast('Laws', compilation.drawn)
        ]
        drawing.allocate()
        compilation.drawing = drawing
