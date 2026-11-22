"""Define the linear term spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ...base import BaseSpec
from ..types import Chain
from ._field_element import _FieldElement
from ._reference_spec import ReferenceSpec


@dataclass(frozen=True, eq=False)
class LinearTermSpec(BaseSpec):
    """A spec of a term of the rate of a field that applies fixed linear operators.

    A rate on a grid often holds a term such as `D * laplacian(u)` for diffusion, a coefficient times a chain of fixed
    linear operators applied to a state. The engine keeps such terms apart, so a solver can treat them implicitly or in
    Fourier space, where they are cheap and stable. It spreads heat in a plate, a chemical in a gel, and a species over
    a landscape.

    Read more in the user guide, under Solvers.

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

    See Also:
        LinearOperatorKind: The kinds of operator in a chain.

    Examples:
        >>> from skcomplex.systems.physics.heat import HeatDiffusionSystem
        >>> term = HeatDiffusionSystem().prepare().simulate(until=0).problem.linear[0]
        >>> term.chain, str(term.coefficient)
        ((('laplacian', 0),), 'D_plate')
    """

    target: ReferenceSpec
    source: ReferenceSpec
    place: _FieldElement
    chain: Chain
    coefficient: Expression
    references: Mapping[Expression, ReferenceSpec]
