"""Pass that takes the rules out of the laws lowered into kernels, since the agent stepper runs them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from ....core.laws import RuleLaw
from .._compilation import Compilation, Laws
from .._pipeline import BasePass


class SetRulesApartPass(BasePass):
    """Pass that takes the rules out of the laws lowered into kernels, since the agent stepper runs them."""

    reads = ('reduced',)
    writes = ('reduced', 'rules')

    def _run(self, compilation: Compilation) -> None:
        """Take the rules out of the reduced laws.

        Args:
            compilation:
                The compilation state.
        """
        reduced = cast('Laws', compilation.reduced)
        compilation.facts['rules'] = [
            (spec, tuple(law for law in laws if isinstance(law, RuleLaw))) for spec, laws in reduced
        ]
        compilation.reduced = [(spec, [law for law in laws if not isinstance(law, RuleLaw)]) for spec, laws in reduced]
