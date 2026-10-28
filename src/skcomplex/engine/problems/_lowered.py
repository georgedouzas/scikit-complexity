"""Kernels of every law kind, with the changes of every jump output and the law of every output."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ...core.laws import BaseLaw
from ._kernel import Kernel
from ._reference import Reference


@dataclass
class Lowered:
    """Kernels of every law kind, with the changes of every jump output and the law of every output.

    Args:
        kernels:
            The kernels of every law kind, in evaluation order.

        jump_changes:
            The state changes of every jump output, keyed by the output's identity.

        laws:
            The kernel spec and the law every output was lowered from, keyed by the output's identity.
    """

    kernels: dict[type, list[Kernel]] = field(default_factory=lambda: defaultdict(list))
    jump_changes: dict[int, list[tuple[Reference, float]]] = field(default_factory=dict)
    laws: dict[int, tuple[Any, BaseLaw]] = field(default_factory=dict)
