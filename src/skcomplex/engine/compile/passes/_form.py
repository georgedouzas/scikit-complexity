"""Infer the form of a problem from the classes of the laws of its components."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from .._compilation import Compilation
from .._forms import infer_form
from .._pipeline import BasePass


class FormPass(BasePass):
    """Pass that infers the form of the problem from the law classes of its owners."""

    reads = ('kinds',)
    writes = ('form',)

    def _run(self, compilation: Compilation) -> None:
        """Infer the form of the problem.

        Args:
            compilation:
                The compilation state.
        """
        compilation.form = infer_form(cast('dict[str, set[type]]', compilation.kinds).values())
