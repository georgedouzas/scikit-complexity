"""Lower the member offset reads of laws to neighbour reads of the index table of their group."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import defaultdict
from dataclasses import replace
from typing import cast

from sage.all__sagemath_symbolics import SR

from ...base import Expression
from ...laws import Boundary, Jump, Law, Noise, Rate, Update
from .._compilation import Compilation, Laws
from .._declarations import Declarations
from .._members import MemberShifts
from .._pipeline import Pass
from .._specs import KernelSpec


def _with_member_shifts(spec: KernelSpec, laws: list[Law], time: Expression) -> list[Law]:
    """Return laws whose member offset reads are lowered to neighbour reads of the group's index table.

    A jump keeps a change whose target is not a symbolic expression, such as a request of another formalism.
    """
    shifts = MemberShifts(spec.owner, spec.references, time)
    result: list[Law] = []
    for law in laws:
        if isinstance(law, Jump):
            changes: dict[Expression, float] = defaultdict(float)
            for target, increment in law.changes.items():
                changes[SR(shifts.lower(target)) if isinstance(target, Expression) else target] += increment
            result.append(Jump(shifts.lower(law.rate), dict(changes)))
        elif isinstance(law, Rate | Noise):
            result.append(replace(law, value=shifts.lower(law.value)))
        elif isinstance(law, Update | Boundary):
            result.append(type(law)(law.target, shifts.lower(law.value)))
        else:
            result.append(law)
    return result


class MemberShiftPass(Pass):
    """Pass that lowers the member offset reads of laws to neighbour reads."""

    reads = ('corrected', 'layout')
    writes = ('shifted',)

    def run(self, compilation: Compilation) -> None:
        """Lower the member offset reads of the corrected laws.

        Args:
            compilation:
                The compilation state.
        """
        time = cast('Declarations', compilation.layout).time
        compilation.shifted = [
            (spec, _with_member_shifts(spec, laws, time)) for spec, laws in cast('Laws', compilation.corrected)
        ]
