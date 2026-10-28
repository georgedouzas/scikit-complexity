"""Split the rates of fields into a linear part with constant coefficients and a nonlinear remainder."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ...core.elements import FieldElement
from ...types import Chain
from ..problems import Reference

OPERATORS = ('laplacian', 'partial', 'inverse_laplacian')


@dataclass(frozen=True, eq=False)
class LinearTerm:
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

    target: Reference
    source: Reference
    place: FieldElement
    chain: Chain
    coefficient: Expression
    references: Mapping[Expression, Reference]
