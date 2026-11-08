"""Define the compiled delay spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ...specs._base_spec import BaseSpec
from ._reference_spec import ReferenceSpec


@dataclass(frozen=True)
class _CompiledDelaySpec(BaseSpec):
    """Delay or smoothing of a compiled system.

    It holds its call in the laws and its output, read from hidden states.
    """

    call: Expression
    output: Expression
    references: Mapping[Expression, ReferenceSpec]
