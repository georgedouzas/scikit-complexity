"""Define the lowered kernels spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ...base import BaseSpec
from ...components.laws import BaseLaw
from ...enums import FormKind
from ._draw_kernel import DrawKernel
from ._kernel_spec import KernelSpec
from ._law_kernel import LawKernel
from ._neighborhood_kernel import _ReductionSchedule
from ._random_stream import RandomStream
from ._reference_spec import ReferenceSpec


@dataclass(frozen=True)
class LoweredKernelsSpec(BaseSpec):
    """A spec of the kernels that the laws of a system compile to, by kind of law.

    After it lowers every law, the compiler holds the kernels of every kind of law in the order they run, the changes of
    every jump, the law every value came from, and the reductions, draws, and random stream the kernels read, such as
    the rate kernel of the growth of a population. Later passes build the problem from it. It holds the compiled laws of
    an epidemic, a reactor, and a market.

    Read more in the user guide, under Extend the library.

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

    See Also:
        LawKernel: One kernel of laws.

    Examples:
        >>> from collections import defaultdict
        >>> from skcomplex.core.engine.problems._lowered_kernels_spec import LoweredKernelsSpec
        >>> LoweredKernelsSpec(defaultdict(list), {}, {}, size=3).size
        3
    """

    kernels: defaultdict[type, list[LawKernel]]
    jump_changes: dict[int, list[tuple[ReferenceSpec, object]]]
    laws: dict[int, tuple[KernelSpec, BaseLaw]]
    form: FormKind | None = None
    size: int = 0
    reductions: _ReductionSchedule | None = None
    draws: DrawKernel | None = None
    stream: RandomStream | None = None
