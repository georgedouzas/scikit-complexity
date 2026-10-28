"""Pass that replaces every jump on a clock of waiting times by the renewal the steppers run for it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from ....core.laws import JumpLaw
from .._compilation import BlockLaws, Compilation
from .._pipeline import BasePass
from .._renewal_law import RenewalLaw


class UnpackRenewalsPass(BasePass):
    """Pass that replaces every jump on a clock of waiting times by the renewal the steppers run for it."""

    reads = ('collected',)
    writes = ('collected',)

    def _run(self, compilation: Compilation) -> None:
        """Replace the jumps on clocks of waiting times by renewals.

        Args:
            compilation:
                The compilation state.
        """
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
