"""Define the hilbert spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..errors import ValidationError
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class HilbertSpec(BaseSpec):
    """A spec of the complex vector space of the states of an element.

    It holds the local operators its laws build generators from. An element that takes a `HilbertSpec` and declares a
    complex state, a `StateSpec` over `CC` or `CC^dimension`, is a subsystem. Its laws read every local operator by
    name, and a group of such elements is the tensor product of its members' spaces, in member order.

    Args:
        dimension:
            The dimension of the space.

        operators:
            A mapping from operator name to its matrix and whether it is Hermitian, such as `{'z': (matrix, True)}`.

        basis:
            A mapping from the label of a basis state to its index, such as `{'up': 0, 'down': 1}`.

        display:
            The name of the operator whose expectation visuals show for a group of such elements, or `None` for the
            probabilities of the basis states.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.specs import HilbertSpec
        >>> spec = HilbertSpec(2, {'z': (np.diag([1.0, -1.0]), True)}, basis={'up': 0, 'down': 1})
        >>> spec.basis['down']
        1
    """

    dimension: int
    operators: Mapping[str, tuple[npt.ArrayLike, bool]] = field(default_factory=dict)
    basis: Mapping[str, int] = field(default_factory=dict)
    display: str | None = 'z'

    def _check(self) -> None:
        """Raise an error for a dimension, operator, basis state, or display that misfits.

        They must fit the space.

        Raises:
            ValidationError:
                The dimension is not a positive integer, an operator is not a square matrix of the dimension, a basis
                index is out of range, or the display operator is not one of the operators.
        """
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension < 1:
            error_msg = f'The dimension of a Hilbert space must be a positive integer, but it is {self.dimension!r}.'
            raise ValidationError(error_msg)
        for name, (matrix, _) in self.operators.items():
            if np.shape(matrix) != (self.dimension, self.dimension):
                error_msg = (
                    f'The operator {name!r} of a Hilbert space of dimension {self.dimension} has the shape '
                    f'{np.shape(matrix)}. Give a square matrix of the dimension.'
                )
                raise ValidationError(error_msg)
        for label, index in self.basis.items():
            if not 0 <= index < self.dimension:
                error_msg = f'The basis state {label!r} has the index {index}, outside the dimension {self.dimension}.'
                raise ValidationError(error_msg)
        if self.display is not None and self.operators and self.display not in self.operators:
            error_msg = f'The display operator {self.display!r} is not one of {", ".join(self.operators)}.'
            raise ValidationError(error_msg)
