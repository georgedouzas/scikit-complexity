"""Pass that takes the rules out of the laws lowered into kernels, since the agent stepper runs them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from ....core.laws import RuleLaw
from .._base import BasePass
from .._compilation import Laws


class SetRulesApartPass(BasePass):
    """Pass that takes the rules out of the laws lowered into kernels, since the agent stepper runs them."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('reduced',)

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('reduced', 'rules')

    def _run(self, inputs: object = None) -> None:
        """Take the rules out of the reduced laws."""
        compilation = self.compilation_
        reduced = cast('Laws', compilation.reduced)
        compilation.rules = [
            (spec, tuple(law for law in laws if isinstance(law, RuleLaw))) for spec, laws in reduced
        ]
        compilation.reduced = [(spec, [law for law in laws if not isinstance(law, RuleLaw)]) for spec, laws in reduced]
