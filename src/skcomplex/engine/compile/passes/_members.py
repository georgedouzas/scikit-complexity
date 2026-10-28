"""Lower the member offset reads of laws to neighbour reads of the index table of their group."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import defaultdict
from typing import cast

from sage.all__sagemath_symbolics import SR, Expression

from ....core.laws import BaseLaw, BoundaryLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from .._base import BasePass
from .._compilation import Laws
from .._declarations import Declarations
from .._kernel_spec import KernelSpec
from .._members import MemberShifts


class MemberShiftPass(BasePass):
    """Pass that lowers the member offset reads of laws to neighbour reads."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('corrected', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('shifted',)

    def _run(self, inputs: object = None) -> None:
        """Lower the member offset reads of the corrected laws."""
        compilation = self.compilation_
        time = cast('Declarations', compilation.layout).time
        compilation.shifted = [
            (spec, self._with_member_shifts(spec, laws, time)) for spec, laws in cast('Laws', compilation.corrected)
        ]

    @staticmethod
    def _with_member_shifts(spec: KernelSpec, laws: list[BaseLaw], time: Expression) -> list[BaseLaw]:
        """Return laws whose member offset reads are lowered to neighbour reads of the group's index table.

        A jump keeps a change whose target is not a symbolic expression, such as a request of another formalism.
        """
        shifts = MemberShifts(spec.owner, spec.references, time)
        result: list[BaseLaw] = []
        for law in laws:
            if isinstance(law, JumpLaw):
                changes: dict[Expression, float] = defaultdict(float)
                for target, increment in law.changes.items():
                    changes[SR(shifts.lower(target)) if isinstance(target, Expression) else target] += increment
                result.append(JumpLaw(shifts.lower(law.rate), dict(changes)))
            elif isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw):
                result.append(BasePass._replaced(law, value=shifts.lower(law.value)))
            else:
                result.append(law)
        return result
