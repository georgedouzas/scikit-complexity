"""Define the lowered kernels spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._base_spec import BaseSpec
from ._reduction_schedule_spec import ReductionScheduleSpec
from ._reference_spec import ReferenceSpec

if TYPE_CHECKING:
    from ..core.laws import BaseLaw
    from ..engine.problems._draw_kernel import DrawKernel
    from ..engine.problems._law_kernel import LawKernel
    from ..types import FormKind


@dataclass(frozen=True)
class LoweredKernelsSpec(BaseSpec):
    """Kernels of every law kind, with the changes of every jump output and the law of every output.

    Args:
        kernels:
            The kernels of every law kind, in evaluation order.

        jump_changes:
            The state changes of every jump output, keyed by the output's identity.

        laws:
            The kernel spec and the law every output was lowered from, keyed by the output's identity.

        form:
            The form of the problem, which a backend writes the numeric functions of, or `None` before it is known.

        size:
            The size of the state vector.

        reductions:
            The reductions the kernels read, evaluated before them at every evaluation, or `None`.

        draws:
            The random values a stochastic map draws at every step into the parameter vector extended after its
            parameters, or `None` for a deterministic problem.
    """

    kernels: defaultdict[type, list[LawKernel]]
    jump_changes: dict[int, list[tuple[ReferenceSpec, float]]]
    laws: dict[int, tuple[object, BaseLaw]]
    form: FormKind | None = None
    size: int = 0
    reductions: ReductionScheduleSpec | None = None
    draws: DrawKernel | None = None
