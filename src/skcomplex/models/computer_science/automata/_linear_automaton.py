"""Linear cellular automaton, an element of a group ring acting on configurations over a field."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import GF, QQ, LaurentPolynomialRing, matrix
from sage.rings.polynomial.laurent_polynomial import LaurentPolynomial
from sage.rings.ring import Field
from sage.structure.element import Matrix

from ....core.neighborhoods import LatticeNeighborhood
from ....core.spaces import BaseSpace
from ....errors import ValidationError
from ._automaton import Reach, _size, _table


@dataclass(frozen=True)
class LinearAutomaton:
    """Linear cellular automaton, the element `sum_s a_s s` of the group ring `K[G]` acting on configurations.

    The value at the site `g` becomes `sum_s a_s x(g s)` over the field, for every element `s` of the neighbourhood
    with its coefficient `a_s`. On a finite universe it is a square matrix, and injectivity, surjectivity, and
    pre-injectivity all hold exactly when the matrix is invertible. On `Z^d` it is a Laurent polynomial, invertible
    when it is a monomial and surjective when it is not zero.

    Args:
        space:
            The periodic lattice or the group space of a finite group or quotient.

        neighborhood:
            The neighbourhood of offsets on a lattice, or the memory set on a group space.

        coefficients:
            The coefficient of every element of the neighbourhood, integers or rationals.

        field:
            `'GF(p)'` for the field of a prime number `p` of elements, or `'QQ'` for the rationals.

    Examples:
        >>> from skcomplex.core.neighborhoods import LatticeNeighborhood
        >>> from skcomplex.core.spaces import LatticeSpace
        >>> from skcomplex.models.computer_science.automata import LinearAutomaton
        >>> rule90 = LinearAutomaton(LatticeSpace(6), LatticeNeighborhood(((-1,), (1,))), (1, 1), 'GF(2)')
        >>> rule90.step([0, 0, 1, 0, 0, 0]).tolist()
        [0, 1, 0, 1, 0, 0]
    """

    space: BaseSpace
    neighborhood: Reach
    coefficients: Sequence[float]
    field: str = 'GF(2)'

    def _field(self) -> Field:
        """Return the Sage field of `'GF(p)'` or `'QQ'`."""
        if self.field == 'QQ':
            return QQ
        text = self.field.replace(' ', '')
        if text.startswith('GF(') and text.endswith(')') and text[3:-1].isdigit():
            return GF(int(text[3:-1]))
        error_msg = f"Unknown field {self.field!r}. Use 'QQ' or 'GF(p)' for a prime p."
        raise ValidationError(error_msg)

    def step(self, configuration: npt.ArrayLike) -> npt.NDArray[np.generic]:
        """Return the image of a configuration.

        Args:
            configuration:
                The values of the sites.

        Returns:
            image:
                The next values, reduced modulo the field size over a prime field.
        """
        table = _table(self.space, self.neighborhood)
        values = np.append(np.asarray(configuration), getattr(self.space, 'fill', 0))
        image = values[table] @ np.asarray(self.coefficients)
        characteristic = int(self._field().characteristic())
        return np.rint(image).astype(np.int64) % characteristic if characteristic else image

    def matrix(self) -> Matrix:
        """Return the matrix of the automaton over the field, whose row `g` gives the image at `g`.

        Returns:
            matrix:
                The Sage matrix.

        Raises:
            ValidationError:
                A site reads a cell outside the space.
        """
        table = _table(self.space, self.neighborhood)
        if (table >= _size(self.space)).any():
            error_msg = 'A linear automaton has a matrix on a periodic lattice or a finite group, not on a ball.'
            raise ValidationError(error_msg)
        field = self._field()
        entries = matrix(field, _size(self.space), _size(self.space))
        for site, row in enumerate(table.tolist()):
            for neighbour, coefficient in zip(row, self.coefficients, strict=True):
                entries[site, neighbour] += field(QQ(coefficient))
        return entries

    def laurent_polynomial(self) -> LaurentPolynomial:
        """Return the Laurent polynomial `sum a_s x^s` of an automaton on a lattice, its symbol on `Z^d`.

        Returns:
            polynomial:
                The Sage Laurent polynomial in the variables `x0` to `x(d-1)`.

        Raises:
            ValidationError:
                The neighbourhood is not one of offsets.
        """
        if not isinstance(self.neighborhood, LatticeNeighborhood):
            error_msg = 'A Laurent polynomial describes a linear automaton on a lattice, with offsets.'
            raise ValidationError(error_msg)
        field = self._field()
        offsets = self.neighborhood.check().prepare().offsets_
        n_dim = len(offsets[0])
        ring = LaurentPolynomialRing(field, [f'x{axis}' for axis in range(n_dim)])
        variables = ring.gens()
        terms = (
            field(QQ(coefficient)) * np.prod([variable**step for variable, step in zip(variables, offset, strict=True)])
            for offset, coefficient in zip(offsets, self.coefficients, strict=True)
        )
        return sum(terms, ring.zero())
