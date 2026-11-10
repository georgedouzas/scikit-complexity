"""Pass that replaces every matching by the rule it runs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....components.laws import MatchLaw
from .._base import BasePass
from .._compilation import Compilation


class UnpackMatchesPass(BasePass):
    """A pass that replaces every matching by the rule it runs."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('collected',)

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('collected',)

    def _run(self, inputs: object = None) -> None:
        """Replace the matchings by their rules."""
        compilation = self.compilation_
        compilation.collected = [
            (spec, [law._get_rule() if isinstance(law, MatchLaw) else law for law in laws], block)
            for spec, laws, block in Compilation._get_required(compilation.collected, 'collected')
        ]
