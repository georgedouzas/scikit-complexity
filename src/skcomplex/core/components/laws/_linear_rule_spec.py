"""Define the linear rule spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

from sage.all__sagemath_symbolics import GF, QQ
from sage.rings.ring import Field

from ...base import BaseSpec
from ...errors import ValidationError


@dataclass(frozen=True)
class LinearRuleSpec(BaseSpec):
    """A spec of a local rule that adds the values of the neighbours with coefficients.

    It gives the next value of a cell as `sum_s a_s x(g s)` over its neighbours `s` with the coefficients `a_s`,
    counted in a finite field or in the rationals, such as `LinearRuleSpec((1, 1))` with the left and right neighbours
    of a cell for rule 90, where every cell becomes the sum modulo 2 of its two neighbours. Over the group ring `K[G]`
    the rule is the element `sum_s a_s s`, so the algebra of the ring describes the patterns it makes. Rule 90 draws a
    Sierpinski triangle, linear codes spread errors, and additive automata draw the stripes of seashells.

    Read more in the user guide, under Laws.

    Args:
        coefficients:
            The coefficient of every neighbour, in neighbourhood order, integers or rationals.

        field:
            `'GF(p)'` for the field of a prime number `p` of elements, or `'QQ'` for the rationals.

    See Also:
        TotalisticSpec: A local rule given by the sum of the neighbours.
        UpdateLaw: Takes the spec as the value of a local rule.

    Examples:
        >>> from skcomplex.core.components.laws import LinearRuleSpec
        >>> LinearRuleSpec((1, 1), 'GF(3)').characteristic
        3
    """

    coefficients: Sequence[float]
    field: str = 'GF(2)'

    def _check(self) -> None:
        """Raise an error for an unknown field.

        Raises:
            ValidationError:
                The field is neither `'QQ'` nor `'GF(p)'`.
        """
        text = self.field.replace(' ', '')
        if text != 'QQ' and not (text.startswith('GF(') and text.endswith(')') and text[3:-1].isdigit()):
            error_msg = f"Unknown field {self.field!r}. Use 'QQ' or 'GF(p)' for a prime p."
            raise ValidationError(error_msg)

    @property
    def sage_field(self) -> Field:
        """Sage field of the coefficients."""
        text = self.field.replace(' ', '')
        return QQ if text == 'QQ' else GF(int(text[3:-1]))

    @property
    def characteristic(self) -> int:
        """Characteristic of the field, 0 for the rationals."""
        return int(self.sage_field.characteristic())
