"""Operator that acts on the Hilbert space of one subsystem, identified by its name."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import qutip
import scipy.sparse as sp

_ADJOINT_NAMES = {'plus': 'minus', 'minus': 'plus', 'annihilate': 'create', 'create': 'annihilate'}


_ADJOINT_SUFFIX = '.dag'


@dataclass(frozen=True)
class LocalOperator:
    """Operator that acts on the Hilbert space of one subsystem, identified by its name.

    Args:
        name:
            The name of the operator, unique among the operators of its subsystem.

        matrix:
            The matrix of the operator.

        hermitian:
            Whether the operator equals its adjoint.

    Examples:
        >>> import qutip
        >>> from skcomplex.engine.compile import LocalOperator
        >>> LocalOperator('plus', qutip.sigmap()).adjoint().name
        'minus'
    """

    name: str
    matrix: Any = field(default=None, compare=False, repr=False)
    hermitian: bool = field(default=False, compare=False)

    def adjoint(self) -> 'LocalOperator':
        """Return the adjoint of the operator.

        Returns:
            adjoint:
                The operator itself when it is Hermitian, or the operator with the conjugate transposed matrix.
        """
        if self.hermitian:
            return self
        if self.name in _ADJOINT_NAMES:
            name = _ADJOINT_NAMES[self.name]
        elif self.name.endswith(_ADJOINT_SUFFIX):
            name = self.name.removesuffix(_ADJOINT_SUFFIX)
        else:
            name = f'{self.name}{_ADJOINT_SUFFIX}'
        matrix = self._sparse_matrix(self.matrix).conj().T.tocsr()
        return LocalOperator(name, matrix, hermitian=False)

    @staticmethod
    def _sparse_matrix(matrix: Any) -> sp.csr_array:  # noqa: ANN401
        """Return a matrix, a QuTiP object, or an array as a complex sparse array."""
        if isinstance(matrix, qutip.Qobj):
            matrix = matrix.full()
        return sp.csr_array(np.asarray(matrix.toarray() if sp.issparse(matrix) else matrix, dtype=complex))
