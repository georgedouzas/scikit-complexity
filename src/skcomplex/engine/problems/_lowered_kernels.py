"""Kernels of every law kind, with the changes of every jump output and the law of every output."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import defaultdict
from dataclasses import dataclass

from ...core.laws import BaseLaw
from ._base import BaseValue
from ._law_kernel import LawKernel
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
    """

    kernels: defaultdict[type, list[LawKernel]]
    jump_changes: dict[int, list[tuple[Reference, float]]]
    laws: dict[int, tuple[object, BaseLaw]]
