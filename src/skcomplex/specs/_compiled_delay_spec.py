"""Define the compiled delay spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class _CompiledDelaySpec(BaseSpec):
    """Delay or smoothing of a compiled system, its call in the laws and its output read from hidden states."""

    call: Expression
    output: Expression
    references: Mapping[Expression, object]
