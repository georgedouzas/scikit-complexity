"""Settle the rows every kernel spec is evaluated over, after checking where its laws draw random values."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from .._compilation import BlockLaws, Compilation, Laws
from .._pipeline import Pass
from .._specs import settle_rows
from .._stochastic import check_draw_placement


class RowsPass(Pass):
    """Pass that checks the draws of the laws and settles the rows of every kernel spec, with its laws set apart."""

    reads = ('lowered', 'apart')
    writes = ('lowered',)

    def run(self, compilation: Compilation) -> None:
        """Check the draws and settle the rows of every kernel spec.

        Args:
            compilation:
                The compilation state.
        """
        apart = cast('Laws', compilation.apart)
        for spec, laws, _ in cast('BlockLaws', compilation.lowered):
            check_draw_placement(spec.owner, laws)
            settle_rows(spec, [*laws, *(law for other, items in apart if other is spec for law in items)])
