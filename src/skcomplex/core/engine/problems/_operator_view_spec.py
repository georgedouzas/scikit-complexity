"""Define the spec of an element whose state lies in a Hilbert space."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ...components.elements import Element
from ...errors import ValidationError
from ...specs._base_spec import BaseSpec
from ..types import HilbertSpaceProtocol
from ._local_operator_spec import LocalOperatorSpec

NORM_TOLERANCE = 1e-8


@dataclass(frozen=True)
class _OperatorViewSpec(BaseSpec):
    """A spec of an element whose state is a vector of a Hilbert space.

    A Hilbert space is the space of the states of a quantum system. The compiler of operators reads this view to build
    the matrices of the evolution.

    Args:
        element:
            The element.

        state:
            The name of the state of the element in its Hilbert space.

        space:
            The Hilbert space of the element.
    """

    element: Element
    state: str
    space: HilbertSpaceProtocol

    @property
    def label(self) -> str:
        """Label of the element."""
        return self.element.label

    @property
    def dimension(self) -> int:
        """Dimension of the Hilbert space."""
        return self.space.dimension

    @property
    def display(self) -> str | None:
        """Operator whose expectation visuals show for a group.

        It is `None` for the probabilities of the basis.
        """
        return self.space.display

    def get_local_operators(self) -> dict[str, LocalOperatorSpec]:
        """Return the local operators of the element by name.

        Returns:
            operators:
                A mapping from name to local operator.
        """
        return {
            name: LocalOperatorSpec(name, LocalOperatorSpec._get_sparse_matrix(matrix), hermitian=hermitian)
            for name, (matrix, hermitian) in self.space.operators.items()
        }

    def get_basis_index(self, label: object) -> int:
        """Return the index of the basis state of a label or an index.

        Args:
            label:
                The label of a basis state, or its index.

        Returns:
            index:
                The index of the basis state.

        Raises:
            ValidationError:
                The label names no basis state.
        """
        space = self.space
        if isinstance(label, str) and label in space.basis:
            return space.basis[label]
        if isinstance(label, int | np.integer) and not isinstance(label, bool) and 0 <= label < space.dimension:
            return int(label)
        labels = ', '.join(map(repr, space.basis)) or 'none'
        error_msg = (
            f'{type(self.element).__name__} {self.label!r}: {label!r} is no basis state. Use a label ({labels}) or an '
            f'index from 0 to {space.dimension - 1}.'
        )
        raise ValidationError(error_msg)

    def get_initial_vector(self) -> npt.NDArray[np.complex128]:
        """Return the initial state of the element as a normalised vector.

        Returns:
            vector:
                The initial state.

        Raises:
            ValidationError:
                The initial state names no basis state, or is a vector of the wrong size or norm.
        """
        initial = getattr(self.element, self.state)
        if isinstance(initial, str | int | np.integer):
            vector = np.zeros(self.dimension, dtype=complex)
            vector[self.get_basis_index(initial)] = 1.0
            return vector
        vector = np.asarray(initial, dtype=complex).ravel()
        norm = float(np.linalg.norm(vector))
        if vector.size != self.dimension or abs(norm - 1) > NORM_TOLERANCE:
            error_msg = (
                f'The initial vector of {self.label!r} has {vector.size} components and norm {norm:.6g}, but its '
                f'space has dimension {self.dimension} and states have norm 1. Normalise the vector.'
            )
            raise ValidationError(error_msg)
        return vector
