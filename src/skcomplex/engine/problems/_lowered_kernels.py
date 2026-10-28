"""Kernels of every law kind of a problem, with what a backend needs to write its numeric functions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import defaultdict
from dataclasses import dataclass

from ...core.laws import BaseLaw
from ._base import BaseValue
from ._draw_kernel import DrawKernel
from ._form import Form
from ._law_kernel import LawKernel
from ._reduction_schedule import ReductionSchedule
from ._reference import Reference


@dataclass(frozen=True)
class LoweredKernels(BaseValue):
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
    jump_changes: dict[int, list[tuple[Reference, float]]]
    laws: dict[int, tuple[object, BaseLaw]]
    form: Form | None = None
    size: int = 0
    reductions: ReductionSchedule | None = None
    draws: DrawKernel | None = None
