"""Split the laws of complex states into laws of their real and imaginary parts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial
from typing import cast

from .._compilation import BlockLaws, Compilation
from .._complex import split_law
from .._pipeline import BasePass
from .._specs import state_name


class LowerComplexPass(BasePass):
    """Pass that splits every law of a complex state into the laws of its real and imaginary parts."""

    reads = ('collected',)
    writes = ('lowered',)

    def run(self, compilation: Compilation) -> None:
        """Split the collected laws of complex states.

        Args:
            compilation:
                The compilation state.
        """
        compilation.lowered = [
            (
                spec,
                [item for law in laws for item in split_law(law, spec.owner, partial(state_name, spec))],
                block,
            )
            for spec, laws, block in cast('BlockLaws', compilation.collected)
        ]
