"""Pass that replaces every jump on a clock of waiting times by the renewal the steppers run for it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from ....core.laws import JumpLaw
from .._base import BasePass
from .._compilation import BlockLaws
from .._renewal_law import RenewalLaw


class UnpackRenewalsPass(BasePass):
    """Pass that replaces every jump on a clock of waiting times by the renewal the steppers run for it."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('collected',)

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('collected',)

    def _run(self, inputs: object = None) -> None:
        """Replace the jumps on clocks of waiting times by renewals."""
        compilation = self.compilation_
        compilation.collected = [
            (
                spec,
                [
                    (
                        RenewalLaw(law.waiting, law.reset, label=law.label).check().prepare(law.owner_)
                        if isinstance(law, JumpLaw) and law.waiting is not None
                        else law
                    )
                    for law in laws
                ],
                block,
            )
            for spec, laws, block in cast('BlockLaws', compilation.collected)
        ]
