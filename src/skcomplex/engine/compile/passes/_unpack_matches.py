"""Pass that replaces every matching by the rule it runs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from ....core.laws import MatchLaw
from .._compilation import BlockLaws, Compilation
from .._pipeline import BasePass


class UnpackMatchesPass(BasePass):
    """Pass that replaces every matching by the rule it runs."""

    reads = ('collected',)
    writes = ('collected',)

    def _run(self, compilation: Compilation) -> None:
        """Replace the matchings by their rules.

        Args:
            compilation:
                The compilation state.
        """
        compilation.collected = [
            (spec, [law._get_rule() if isinstance(law, MatchLaw) else law for law in laws], block)
            for spec, laws, block in cast('BlockLaws', compilation.collected)
        ]
