"""Define the noise term spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sage.all__sagemath_symbolics import Expression

from ._base_spec import BaseSpec

if TYPE_CHECKING:
    from ..core.laws import NoiseLaw
    from ._kernel_spec import KernelSpec


@dataclass(frozen=True)
class _NoiseTermSpec(BaseSpec):
    """Amplitude of a noise law on one state component, with the process it draws."""

    spec: KernelSpec
    law: NoiseLaw
    target: Expression
    state: tuple[str, int]
    amplitude: object
    process: object
