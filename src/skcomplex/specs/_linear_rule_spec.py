"""Define the linear rule spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

from sage.all__sagemath_symbolics import GF, QQ
from sage.rings.ring import Field

from ..errors import ValidationError
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class LinearRuleSpec(BaseSpec):
    """Rule given by the linear combination `sum_s a_s x(g s)` of the neighbourhood values over a field.

    Over the group ring `K[G]` the rule is the element `sum_s a_s s`, for every element `s` of the neighbourhood with
    its coefficient `a_s`.

    Args:
        coefficients:
            The coefficient of every neighbour, in neighbourhood order, integers or rationals.

        field:
            `'GF(p)'` for the field of a prime number `p` of elements, or `'QQ'` for the rationals.

    Examples:
        >>> from skcomplex.specs import LinearRuleSpec
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
