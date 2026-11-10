"""Define the linear term spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ...components.elements import FieldElement
from ...specs._base_spec import BaseSpec
from ..types import Chain
from ._reference_spec import ReferenceSpec

OPERATORS = ('laplacian', 'partial', 'inverse_helmholtz')


@dataclass(frozen=True, eq=False)
class LinearTermSpec(BaseSpec):
    """A spec of a term of the rate of a field that applies operators to a state.

    The term is multiplied by a constant coefficient.

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
