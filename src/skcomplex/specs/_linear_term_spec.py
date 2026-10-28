"""Define the linear term spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sage.all__sagemath_symbolics import Expression

from ..types import Chain
from ._base_spec import BaseSpec
from ._reference_spec import ReferenceSpec

if TYPE_CHECKING:
    from ..core.elements import FieldElement

OPERATORS = ('laplacian', 'partial', 'inverse_laplacian')


@dataclass(frozen=True, eq=False)
class LinearTermSpec(BaseSpec):
    """Term of the rate of a field state that applies a chain of operators to a state, times a constant coefficient.

    Args:
        target:
            The component whose rate holds the term.

        source:
            The component the operators act on.

        place:
            The field of the source.

        chain:
            The operators from outermost to innermost, with their axes.

        coefficient:
            The coefficient, an expression of parameters.

        references:
            The references of the coefficient's symbols.
    """

    target: ReferenceSpec
    source: ReferenceSpec
    place: FieldElement
    chain: Chain
    coefficient: Expression
    references: Mapping[Expression, ReferenceSpec]
