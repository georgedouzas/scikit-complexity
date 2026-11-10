"""Define the power series spec.

It is the ring of formal power series over a field truncated at an order.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import GF, QQ, RR, PowerSeriesRing
from sage.rings.power_series_ring import PowerSeriesRing_generic
from sage.rings.power_series_ring_element import PowerSeries
from sage.rings.ring import Field
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ..errors import ValidationError
from ..types import Array
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class PowerSeriesSpec(BaseSpec):
    """Ring `K[[t]]` of formal power series over a field, truncated at an order.

    It is the domain of a state of a cell. A state of the ring holds every series `sum_i a_i t^i` by its coefficients
    `a_0, ..., a_{order - 1}`, so it has `order` components, and local rules read the coefficients of every neighbour on
    a last axis. Sums and differences are those of the coefficients, reduced modulo the characteristic, and the product
    is the Cauchy product truncated at the order, so the coefficients up to the order are exact. Sage's power series
    ring of the same precision, `ring`, converts between the coefficients and its series.

    Args:
        order:
            The number of coefficients kept, the precision of the truncation.

        field:
            `'QQ'` for the rationals, `'RR'` for the reals, or `'GF(p)'` for the field of a prime number `p` of
            elements.

        variable:
            The name of the indeterminate.

    Raises:
        ValidationError:
            The order is not a positive integer, or the field is unknown.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.specs import PowerSeriesSpec
        >>> series = PowerSeriesSpec(4)
        >>> series.multiply(np.array([1, 1, 0, 0]), np.array([1, -1, 0, 0])).tolist()
        [1.0, 0.0, -1.0, 0.0]
        >>> series.get_series_of([1, 2, 0, 0])
        1 + 2*t + O(t^4)
        >>> series.get_coefficients_of(series.ring.gen() ** 2).tolist()
        [0.0, 0.0, 1.0, 0.0]
    """

    order: int = 8
    field: str = 'QQ'
    variable: str = 't'

    def _check(self) -> None:
        """Raise an error for an order that is not a positive integer or an unknown field.

        Raises:
            ValidationError:
                The order is not a positive integer, or the field is neither `'QQ'`, `'RR'`, nor `'GF(p)'`.
        """
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 1:
            error_msg = f'The order of a truncated power series ring is a positive integer, but it is {self.order!r}.'
            raise ValidationError(error_msg)
        text = self.field.replace(' ', '')
        if text not in {'QQ', 'RR'} and not (text.startswith('GF(') and text.endswith(')') and text[3:-1].isdigit()):
            error_msg = f"Unknown field {self.field!r}. Use 'QQ', 'RR', or 'GF(p)' for a prime p."
            raise ValidationError(error_msg)

    @property
    def sage_field(self) -> Field:
        """Sage field of the coefficients."""
        text = self.field.replace(' ', '')
        if text == 'QQ':
            return QQ
        if text == 'RR':
            return RR
        return GF(int(text[3:-1]))

    @property
    def alphabet(self) -> object:
        """Sage set of the stored values of a coefficient.

        It is the field or the residues `0` to `p - 1` of a prime field.
        """
        if self.characteristic:
            return FiniteEnumeratedSet(list(range(self.characteristic)))
        return self.sage_field

    @property
    def characteristic(self) -> int:
        """Characteristic of the field, 0 for the rationals and the reals."""
        return int(self.sage_field.characteristic())

    @property
    def ring(self) -> PowerSeriesRing_generic:
        """Sage ring of power series over the field with the precision of the order."""
        return PowerSeriesRing(self.sage_field, self.variable, default_prec=self.order)

    def reduce(self, coefficients: npt.ArrayLike) -> Array:
        """Return coefficients reduced modulo the characteristic.

        They are unchanged over a field of characteristic 0.

        Args:
            coefficients:
                The coefficients on the last axis.

        Returns:
            coefficients:
                The reduced coefficients, as floats.
        """
        values = np.asarray(coefficients, dtype=float)
        return np.mod(np.rint(values), self.characteristic) if self.characteristic else values

    def multiply(self, first: npt.ArrayLike, second: npt.ArrayLike) -> Array:
        """Return the product of two series truncated at the order.

        The series are coefficient arrays on the last axis.

        Args:
            first:
                The coefficients of the first factor, whose leading axes broadcast with those of the second.

            second:
                The coefficients of the second factor.

        Returns:
            product:
                The coefficients of the product, `sum_{j <= i} a_j b_{i - j}` at every power `i` below the order.
        """
        first_values = np.asarray(first, dtype=float)
        second_values = np.asarray(second, dtype=float)
        shape = np.broadcast_shapes(first_values.shape[:-1], second_values.shape[:-1])
        product = np.zeros((*shape, self.order))
        for power in range(self.order):
            product[..., power:] += first_values[..., power, None] * second_values[..., : self.order - power]
        return self.reduce(product)

    def get_series_of(self, coefficients: npt.ArrayLike) -> PowerSeries:
        """Return the series of the ring with some coefficients.

        Args:
            coefficients:
                The coefficients `a_0, ..., a_{order - 1}`.

        Returns:
            series:
                The series of `ring`, with the precision of the order.
        """
        field = self.sage_field
        values = [field(int(item)) if self.characteristic else field(item) for item in np.ravel(coefficients)]
        return self.ring(values, prec=self.order)

    def get_coefficients_of(self, series: object) -> Array:
        """Return the coefficients of a series of the ring up to the order.

        The series can also be anything the ring converts.

        Args:
            series:
                A series, a polynomial in the indeterminate, or a number.

        Returns:
            coefficients:
                The coefficients `a_0, ..., a_{order - 1}`, as floats.
        """
        element = self.ring(series)
        values = element.padded_list(self.order)
        return np.asarray([float(int(item)) if self.characteristic else float(item) for item in values])
