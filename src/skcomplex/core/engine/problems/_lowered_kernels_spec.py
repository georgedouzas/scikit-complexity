"""Define the lowered kernels spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ...components.laws import BaseLaw
from ...specs._base_spec import BaseSpec
from ...types import FormKind
from ._draw_kernel import DrawKernel
from ._kernel_spec import KernelSpec
from ._law_kernel import LawKernel
from ._random_stream import RandomStream
from ._reduction_schedule_spec import ReductionScheduleSpec
from ._reference_spec import ReferenceSpec


@dataclass(frozen=True)
class LoweredKernelsSpec(BaseSpec):
    """A spec of the kernels of every law kind of a problem.

    They come with the changes of every jump output and the law of every output.

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

        stream:
            The random stream the functions of the laws receive, or `None`.
    """

    kernels: defaultdict[type, list[LawKernel]]
    jump_changes: dict[int, list[tuple[ReferenceSpec, object]]]
    laws: dict[int, tuple[KernelSpec, BaseLaw]]
    form: FormKind | None = None
    size: int = 0
    reductions: ReductionScheduleSpec | None = None
    draws: DrawKernel | None = None
    stream: RandomStream | None = None
